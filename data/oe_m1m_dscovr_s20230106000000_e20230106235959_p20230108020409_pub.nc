CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230106000000_e20230106235959_p20230108020409_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-08T02:04:09.236Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-06T00:00:00.000Z   time_coverage_end         2023-01-06T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill         �   records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxXF	@  T          @�{���E@��Bv�Cl{����p�@�Q�B<z�Cw
                                    BxXF�  �          @�z��l(��(�@׮Bt  C=ff�l(��\)@�G�BZ��CV�                                    BxXF&�  �          @�
=���H?���@�z�BK�RC&����H�
=@�\)BO��C:�                                    BxXF52  T          @���tzᾊ=q@�=qBr�
C8)�tz��(�@�
=B_
=CQٚ                                    BxXFC�  �          @��
�Y��?�@�Q�B(�C*n�Y����
=@�z�Bw=qCJ�                                    BxXFR~  T          @���>�  �Mp�@��HB�8RC�0�>�  ���\@���BE  C�b�                                    BxXFa$  T          @��
=�\)�X��@�G�B���C���=�\)��\)@���B?�RC�h�                                    BxXFo�  �          @���>�\)�U�@��B�
=C�b�>�\)��@���B@�RC��=                                    BxXF~p  �          @���?���c�
@ۅBx��C���?����33@��\B8�C�,�                                    BxXF�  �          @�{>�  �AG�@�=qB��=C�` >�  ��z�@�BI�
C�xR                                    BxXF��  �          @��R���(Q�@�{B��
C��=������@��BVQ�C��                                    BxXF�b  T          @�p���R��(�@�B��3C|c׿�R�}p�@�Q�Bi��C��                                    BxXF�  �          @�{� ���@  @أ�Bw�Cl0�� ������@���B=�
Cw33                                    BxXFǮ  �          @�
=��33��@�G�B�p�Cc�\��33�~{@�BZ{Ctp�                                    BxXF�T  
�          A Q�>Ǯ@�\@�p�B�G�B��q>Ǯ=L��A   B�ff@߮                                    BxXF��  �          A��?�R?�{@��B�  B��?�R��
=A ��B�ffC���                                    BxXF�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXGF              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXG�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXG�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXG.8            A��?��\@�  @�  BJ�B��H?��\@C�
@���B��HB�                                      BxXG<�  �          @�
=?�Q�@xQ�@�{Bj�B��
?�Q�?�{@�Q�B�.BeQ�                                    BxXGK�  �          @��R?�  ?��@�\B�Ba�
?�  ����@��B�G�C�+�                                    BxXGZ*  �          @�=q?O\)?�@�\B��Bq�\?O\)�@  @�p�B��C���                                    BxXGh�  �          @��?\)>\@��
B�z�B	�\?\)��
=@�B�=qC�1�                                    BxXGwv  �          @�(�?   �8Q�@�33B��C�4{?   �{@�Q�B�
=C�e                                    BxXG�  T          @�(�>��þ��@�B���C�33>����(�@�ffB���C��                                    BxXG��  �          @��>.{��@�B���C�#�>.{�(Q�@�G�B���C��                                    BxXG�h  T          @�=�G�����@�\B��qC���=�G���H@�{B��RC�@                                     BxXG�  
�          @��k���\@���B��C�N�k��l(�@��
Bo��C�7
                                    BxXG��  T          @��;B�\��(�@�(�B���C�C׾B�\�w�@ə�Bi�C���                                    BxXG�Z  �          @��(��{@�ffB�C�R�(�����@�  BXG�C���                                    BxXG�   �          @�=q��J=q@���Bop�Cl�=���33@���B6��Cv��                                    BxXG�  �          @�  ���H�Q�@��B���Cnh����H��ff@�\)BU�Cz�                                    BxXG�L  "          @��H���
��{@��HB���C��콣�
�s�
@�G�Bo{C�l�                                    BxXH	�  T          @��
�#�
�ٙ�@�p�B�33C�aH�#�
�k�@��BtQ�C��
                                    BxXH�  �          @�׽��Ϳ���@��HB�k�C�E�����c�
@ӅBv�
C�9�                                    BxXH'>  T          @��ý#�
���@�(�B���C�  �#�
�W�@ָRB}=qC���                                    BxXH5�  �          @�׽u��  @陚B��qC��\�u�\��@�33By��C���                                    BxXHD�  T          @�=q    ��{@�  B�C�"�    ���@�B��C�                                    BxXHS0  �          @�
=    ��(�@�33B�\)C��f    �U�@�RB�z�C��
                                    BxXHa�  
�          AG����
�&ffA�B��HCK)���
�6ff@�(�B�  Cqٚ                                    BxXHp|  �          A����
=���RA33B��
C@)��
=�"�\@���B�
=Cp�R                                    BxXH"  �          A����p���A
=B�B�CGc׿�p��0  @��B�\Cq��                                    BxXH��  �          A���A�B�\CA�q��-p�@��
B���ChL�                                    BxXH�n  �          A����
��A (�B�\C@�\��
�,(�@��B��RCec�                                    BxXH�  �          A���
=q�k�@�ffB��CK��
=q�C�
@�z�B|�Cj�=                                    BxXH��  �          @�p������
@�=qB�u�C}�=���q�@ٙ�Br��C�R                                    BxXH�`  �          @��?n{���H@�Q�B���C��R?n{�w�@�ffBiG�C��=                                    BxXH�  �          @��R?���%�@�
=B�ffC�33?�����
@�  BR\)C��{                                    BxXH�  �          @�  >u��33@�ffB�(�C���>u�vff@���Bop�C���                                    BxXH�R  �          A ��<��
�fff@��RB�{C���<��
�A�@��B��{C�0�                                    BxXI�  �          @��?
=q�\)@�33B��qC�'�?
=q��z�@���B]Q�C���                                    BxXI�  T          @��>�  ��33@��HB�aHC�!H>�  �^�R@�p�B�(�C�
                                    BxXI D  �          A (��8Q�z�H@��B�� C�� �8Q��Dz�@陚B�B�C�N                                    BxXI.�  �          @�z����=q@���B�C��=���H��@�{B��qC��R                                    BxXI=�  "          @��
��G�����@�\)B�\C��=��G��X��@�\B�ǮC��                                    BxXIL6  �          @�(��޸R=�G�@�z�B�W
C0aH�޸R����@�z�B�CdaH                                    BxXIZ�  �          @��H�J=q���@��RB��=Ci0��J=q�Fff@�z�B�ǮC��                                    BxXIi�  �          @�=q���L��@��B�8RCn޸���6ff@�B�k�C�0�                                    BxXIx(  �          @��ÿO\)���R@�\B���Cqz�O\)�^{@���Bz��C�p�                                    BxXI��  �          @��ÿ�\��@�  B�B�CͿ�\�w�@ָRBoQ�C�7
                                    BxXI�t  �          @�=q>�=q�#33@��B�\)C���>�=q��p�@˅B\�RC���                                    BxXI�  �          @�{?�33��Q�@�ffBfz�C��
?�33���@�(�B*�HC��                                    BxXI��  T          A�?�
=�~{@ٙ�Bd
=C��?�
=����@��B*�C�h�                                    BxXI�f  T          @��R@33��{@�ffBXz�C�@33��Q�@�33BG�C��\                                    BxXI�  "          A (�?�
=��Q�@�  BY(�C�(�?�
=���H@�(�BffC�                                      BxXI޲  �          @�(�@�����H@��BN�HC�t{@����=q@�G�B=qC�\)                                    BxXI�X  "          @�z�?��R��Q�@�(�BL��C��=?��R��\)@��RB  C�8R                                    BxXI��  �          @�?޸R��{@ʏ\BT�C��{?޸R���R@�BC�"�                                    BxXJ
�  T          @�\)?�G��p  @��Bp�
C�K�?�G���ff@�p�B5�\C��                                    BxXJJ  "          @�\)?�����
@�(�B`�
C��{?����\)@���B&��C��R                                    BxXJ'�  T          @�?޸R�z=q@�z�Bd��C��)?޸R����@��
B+�C��)                                    BxXJ6�  �          @�\)?��g
=@޸RBs�C���?���=q@�Q�B933C�z�                                    BxXJE<  �          A z�?��
�c33@���Bt��C��?��
����@�33B;=qC��                                    BxXJS�  �          Ap�?���W�@�B}Q�C��H?����z�@�G�BC  C��                                    BxXJb�  "          A��?����H��@�B��{C��H?�����  @���BJ�\C��                                     BxXJq.  
�          A  ?���<(�@�=qB�33C��q?����z�@أ�BT��C�}q                                    BxXJ�  T          Az�?�33�*�HA   B�\C���?�33��p�@��B^=qC��                                    BxXJ�z  "          Az�?���A�B�C��?����(�@�
=BhG�C���                                    BxXJ�   
�          A  ?����{AB��fC�|)?�����Q�@�  Bj�
C�j=                                    BxXJ��  "          A(�?��H�a�@���B�\C�s3?��H����@θRBEC�L�                                    BxXJ�l  T          A=q?   ��=qA��B��HC��\?   �h��@��B~��C��                                    BxXJ�  	�          A
=?#�
����A  B�z�C��f?#�
�l(�@�B~�\C���                                    BxXJ׸  �          A
=?Y���:=q@�33B��3C�'�?Y����33@�=qBX��C���                                    BxXJ�^  T          A{?5�2�\@���B�\)C�
?5���R@�G�B[�RC�AH                                    BxXJ�  
�          A
=?�{��p�A (�B�W
C�\?�{�o\)@�G�Bup�C��{                                    BxXK�  e          A  ?�
=�{@�  B�8RC�t{?�
=���\@�p�Bd=qC��\                                    BxXKP  �          A
{?\(���  @���BO�\C���?\(���=q@�{BQ�C���                                    BxXK �  �          A
�\?��H�tz�@�(�BwG�C�|)?��H���@�z�B>G�C�@                                     BxXK/�            A
ff?��R���R@�  Bd(�C�8R?��R��p�@��B*C��                                    BxXK>B  
�          A
�\?������R@�p�B`  C��R?�����z�@�
=B%��C��                                    BxXKL�  "          A
=q?^�R��33@�z�BT\)C���?^�R��ff@�33B��C��
                                    BxXK[�  
�          A�?�  ��(�@�
=BT�\C�� ?�  �׮@�p�B
=C�4{                                    BxXKj4  �          A  ?k���\)@�{BRQ�C��R?k��ڏ\@�(�B��C��3                                    BxXKx�  T          A\)?�p���{@ۅBP��C���?�p���Q�@��B�C�%                                    BxXK��  �          A	��?�z���p�@��BK{C���?�z���ff@��
BG�C��R                                    BxXK�&  �          A	��?˅��{@ǮB:Q�C���?˅���H@��HB(�C�P�                                    BxXK��  "          A
ff?=p���  @��
B"�C�:�?=p����R@r�\A�ffC���                                    BxXK�r  �          A
ff=#�
��  @��B�C�&f=#�
� ��@HQ�A��C�!H                                    BxXK�  T          A
{?�  ��Q�@�z�B7�\C�H�?�  ��(�@�\)A�
=C��\                                    BxXKо  
�          A	G�?��H���R@θRBD��C�S3?��H��p�@�Bz�C�33                                    BxXK�d  
�          A
�R@'���@�BU�C�C�@'���G�@��HB!\)C��
                                    BxXK�
  "          A
�R?��H��  @���BNz�C���?��H�љ�@��B\)C�T{                                    BxXK��  �          A
{@{��33@��HBGQ�C���@{���H@�33B  C�S3                                    BxXLV  "          A	��@�����@�\)BDG�C��\@���У�@�Q�B�
C�O\                                    BxXL�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXL(�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXL7H              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXLE�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXLT�            AG�>���У�@���B(z�C�  >������@�33A�(�C��                                    BxXLc:  �          A�=�\)��Q�@�(�B6=qC�L�=�\)����@�(�A�C�>�                                    BxXLq�  T          A�\?��R���
@�z�BVQ�C��3?��R��
=@���B\)C�<)                                    BxXL��  �          A33@�R����@��HBP�C��3@�R�Ӆ@�(�B��C�L�                                    BxXL�,  �          A�R?��H����@ۅBA\)C�XR?��H��G�@�
=B
C��H                                    BxXL��  T          A=q@���=q@�
=B?�C��@���33@��\B
ffC��                                    BxXL�x  �          A  @���{@�z�B/�C���@����@�A�{C��                                    BxXL�  "          AG�@\)�Ϯ@�G�B)��C���@\)���\@�Q�A��
C��q                                    BxXL��  T          A@{���
@�Q�B'�\C�J=@{��{@��RA�
=C���                                    BxXL�j  �          A\)?����Q�@љ�B5\)C�U�?����p�@��\A��HC��                                    BxXL�  T          A�H@C�
��{@��B#��C�'�@C�
��R@���A��C�'�                                    BxXL��  �          Aff?�������@�Q�B�
C��?�����\)@|(�A�
=C��q                                    BxXM\  T          A�?���\)@�
=BM=qC�L�?����@�B�C�P�                                    BxXM  �          Az�@�
��z�@�z�B?Q�C���@�
��(�@�Q�B	��C�Ф                                    BxXM!�  
�          A�\?�\)���@�33BK��C��)?�\)��ff@���B�C�n                                    BxXM0N  �          A?Tz�����@�ffB:�RC��?Tz�����@��B�C�/\                                    BxXM>�  �          A�?Y����@�Bz�C�y�?Y��� z�@g
=A���C�                                      BxXMM�  
�          A�H?��
��Q�@���BG�C�#�?��
��@\(�A�  C�Z�                                    BxXM\@  
�          A�\?���(�@�\)B�C�Ф?�� ��@z�HA���C���                                    BxXMj�  T          A�?��
��33@��A�ffC���?��
�z�@7
=A�{C��                                    BxXMy�  "          A
=?�Q���\)@�
=B{C�?�Q�� Q�@Z=qA���C�˅                                    BxXM�2  T          A\)?���ə�@���B*ffC�Z�?����@�(�A��HC�f                                    BxXM��  
�          A�?L����p�@��B7\)C��3?L����Q�@���B ��C��                                    BxXM�~  �          A�
@*�H���@��\B{C�k�@*�H���R@B�\A�33C�Ff                                    BxXM�$  �          A33@0  ��  @�A��
C�e@0  �p�@&ffA�33C�e                                    BxXM��  �          A�
@�����@�  Aә�C�}q@���
@�
ATz�C��                                     BxXM�p  "          A33@h�����H@�B{C��3@h����Q�@���A�C���                                    BxXM�  �          A�@L(����R@�  B7=qC�"�@L(��ڏ\@�G�B=qC���                                    BxXM�  "          A�R@�����@�ffB9�C�ff@���  @��RBffC�^�                                    BxXM�b  
�          A�?����Q�@�  BL�HC��R?���߮@���B�
C�o\                                    BxXN  "          A�\>�=q���HA ��BvG�C���>�=q���H@��B@��C�B�                                    BxXN�  �          A  =��
�mp�A33B���C�� =��
��=q@�BS�C�j=                                    BxXN)T  
�          A(�=��
��ffA{B���C��q=��
�W�A	G�B��C��R                                    BxXN7�  �          A녾���2�\A
=qB��)C��{�����ff@�Q�BjC���                                    BxXNF�  �          A  �B�\�*�HA��B�L�C��B�\��(�A   Bo�\C��\                                    BxXNUF  �          A  ��ff���Az�B�L�C]c׿�ff�K�Az�B�{Cw��                                    BxXNc�  �          A33�W
=��A�
B�� Csk��W
=�k�A�B�ffC���                                    BxXNr�  T          A�\�@  ��ffA�B�\Cw���@  �qG�A�HB��3C�ff                                    BxXN�8  T          A=q�У��A�A�
B���Cq�f�У���=q@�\B\�HC{G�                                    BxXN��  �          AQ�Y��@�G�@�33B\�
B�33�Y��@O\)A��B�{B�z�                                    BxXN��  �          A�?�\>��HA��B���B/ff?�\���HA33B�8RC��                                    BxXN�*  �          Aff?���*=qA
�HB���C���?������@��Bm33C�h�                                    BxXN��  "          A��?(��xQ�A(�B�.C�|)?(���(�@�p�BNC��                                    BxXN�v  �          A=q?@  ��G�@��B
=C��)?@  �33@��AƸRC�aH                                    BxXN�  �          A%�>���=q@�=qB{C���>�����@��HA���C�q�                                    BxXN��  �          A%?
=��@��HB�C�f?
=�Q�@�=qA��HC���                                    BxXN�h  �          A%>��
=@��
Bp�C���>���@b�\A�{C�Y�                                    BxXO  T          A$(���Q��	p�@��Bz�C��콸Q���@c�
A�Q�C���                                    BxXO�  �          A#���=q���@��HB��C�)��=q��@c�
A���C�4{                                    BxXO"Z  �          A%�>�=q��@��HA�ffC�޸>�=q�{@>�RA�ffC�˅                                    BxXO1   �          A"�H>�G���@�G�A�(�C�h�>�G���@,(�Au�C�J=                                    BxXO?�  �          A
=?^�R�
=@k�A�{C��
?^�R�Q�?�=qA{C��                                    BxXONL  �          A{?�(��=q@e�A��C�˅?�(��33?��RA
ffC��3                                    BxXO\�  �          A!�?���\)@��A�C�u�?����@�A_
=C�0�                                    BxXOk�  �          A"{>����\@�  BQ�C��=>���Q�@��\A�{C�y�                                    BxXOz>  !          A%��?����p�@�  A��HC�?�����\@*=qAn�RC�t{                                    BxXO��  
�          A'�?�
=���@�
=A�Q�C�]q?�
=�!p�@AN�RC��                                    BxXO��  T          A&=q?��(�@p  A��C�xR?��!p�?�{A
=C�(�                                    BxXO�0  T          A#33@���\)@HQ�A��HC��=@����\?��\@��C�@                                     BxXO��  T          A*�\@�p�@UA�(�C��3@�%�?�33@ƸRC�e                                    BxXO�|  �          A+33@���@\��A�z�C��\@�%?�G�@���C�^�                                    BxXO�"  T          A+�?�33�
=@�p�A�33C���?�33�%@ ��A,��C�/\                                    BxXO��  T          A.=q?����@���A�{C�y�?���'\)@ffAG
=C��                                    BxXO�n  "          A-��?�(���@�Q�A�Q�C��?�(��'\)@ffAG�C���                                    BxXO�  
�          A2{?\�\)@�\)A�33C�Y�?\�+�@!G�AQp�C�
=                                    BxXP�  	�          A1�?\�{@��
AϮC�^�?\�+
=@+�A^�HC��                                    BxXP`  "          A1��?�33�@�z�A��C��?�33�*�R@-p�Ab=qC���                                    BxXP*  T          A0��?�G���@��A�(�C��3?�G��)�@:=qAt��C�h�                                    BxXP8�  �          A*=q?}p��z�@�33A�\)C��?}p��"=q@C�
A��C�˅                                    BxXPGR  T          A0��?����p�@�G�A�
=C�Ff?����%p�@n{A���C���                                    BxXPU�  
�          A1p�?c�
��@��\A�(�C��R?c�
�&{@p��A��HC�t{                                    BxXPd�  
Z          A/
=?L���ff@���A��
C�n?L���%p�@_\)A���C�5�                                    BxXPsD  
�          A(��?Q��@�  A��HC���?Q���
@P��A�(�C�W
                                    BxXP��  
�          A'�?����@�(�B{C���?��z�@l��A���C��=                                    BxXP��  �          A)>��G�@��B{C�z�>��p�@{�A��C�S3                                    BxXP�6  
�          A(z�?
=q��@�(�BC��f?
=q��@~�RA���C��
                                    BxXP��  "          A(��?W
=��@���B�
C�� ?W
=��@�  A�=qC�w
                                    BxXP��  
�          A(Q�?L����@��HB p�C��
?L����@k�A�C�W
                                    BxXP�(  �          A((�?=p���@�p�B�C�h�?=p��z�@qG�A�(�C�+�                                    BxXP��  �          A&�R?\)�	p�@�33B�C���?\)�p�@�  A��RC���                                    BxXP�t  T          A'\)?+��
=q@��B=qC�4{?+��{@�Q�A�=qC���                                    BxXP�  
�          A&�\?&ff�	�@��B�\C�#�?&ff���@~{A�33C��                                    BxXQ�  
�          A(z�?5��@�ffB(�C�N?5�Q�@u�A��RC�{                                    BxXQf  �          A&�H?����\)@��HA��C�C�?������@Mp�A�  C���                                    BxXQ#  T          A&ff?G��{@��HA���C���?G��(�@^�RA��RC�J=                                    BxXQ1�  
�          A'���\��\@�Q�B33C�5ÿ�\�z�@���A�C�l�                                    BxXQ@X  T          A&�H��\)���@��HB$33C��
��\)��H@�{A�
=C��                                     BxXQN�  �          A)����H����@�ffB$ffC�Ф���H��
@�G�A�ffC��                                     BxXQ]�  T          A(�׿�Q�����@�B$�\C��f��Q��  @���A��\C�u�                                    BxXQlJ  
�          A)G��G�����@�z�B)�HC~�3�G��z�@�G�A��HC��H                                    BxXQz�  "          A(  ��p���33@�\)B.33C~��p��	@�B�HC���                                    BxXQ��  
Z          A&�R����ff@��B8C~�����Q�@�G�B�RC�t{                                    BxXQ�<  "          A%녿�G���=q@�ffB0p�C�)��G��	�@��B�RC��\                                    BxXQ��  
(          A ��������H@��
B==qCx�)�����(�@���B��C|33                                    BxXQ��  "          A��7���=q@��BE\)Cs�7����@��B33Cx#�                                    BxXQ�.  �          A*ff?s33��@��RA�\)C��?s33�@j=qA�Q�C���                                    BxXQ��  �          A1�@ff�33@��RA�ffC��@ff�'�@B�\A~{C��=                                    BxXQ�z  T          A2�H?��H��@��AׅC���?��H�)p�@G�A��
C��
                                    BxXQ�   T          A1�?�
=��@�
=A��C��
?�
=�&�H@Tz�A�C���                                    BxXQ��  
�          A0��?�  �ff@�(�A��HC�"�?�  �'
=@N�RA�  C��                                     BxXRl  
�          A/�@G��z�@���A���C��R@G��'33@'�A]��C�z�                                    BxXR  �          A/�?�ff��\@�33A�{C�E?�ff�&=q@>{A|  C���                                    BxXR*�  T          A/�?�{���@���A��C��f?�{�%p�@S33A��
C�j=                                    BxXR9^  �          A.�\?�p��z�@�  A�=qC�'�?�p��$��@J=qA��RC���                                    BxXRH  2          A.ff?����@�p�A��C���?���$  @VffA�33C�E                                    BxXRV�  
�          A-��?�(���@��A�ffC�j=?�(��$  @P  A�33C�
                                    BxXReP  D          A,��?��
��R@�(�A���C�� ?��
�#
=@U�A��C��{                                    BxXRs�  2          A*�H?����  @�ffA陚C�g�?���� ��@\(�A��RC�"�                                    BxXR��            A+
=?�����@�(�A�\)C��?��� ��@W�A�
=C��                                    BxXR�B  
8          A+\)?�33���@�p�A��C��f?�33�!G�@Z�HA�
=C�AH                                    BxXR��  
�          A,z�?n{���@�G�A��HC��R?n{�"=q@a�A��HC���                                    BxXR��  
(          A,��>�(���@��RA���C�T{>�(��"=q@n{A��HC�9�                                    BxXR�4  "          A,��?p���p�@���A���C�޸?p���"{@b�\A��C��f                                    BxXR��  
�          A.ff?��
�Q�@��AݮC���?��
�$(�@S33A�G�C�=q                                    BxXRڀ  �          A.ff?�{�
=@�ffA�(�C��R?�{�#\)@]p�A�=qC�|)                                    BxXR�&  T          A/�?�{��R@��
A�RC�ٚ?�{�#�@h��A�33C�z�                                    BxXR��  "          A/�
?����@��A�\)C��R?��!�@w
=A�z�C�&f                                    BxXSr  T          A-�?������@�=qA�{C�n?����!�@hQ�A�
=C�
                                    BxXS  
�          A.=q?�z���@�{A�(�C��H?�z��!�@`  A�{C�U�                                    BxXS#�  	�          A/�@	����
@��\Aܣ�C�^�@	���#�@W�A�G�C���                                    BxXS2d  "          A0Q�@
=�z�@��A��HC�AH@
=�$(�@Y��A�C���                                    BxXSA
  
�          A/\)?�G���@��RA��C�B�?�G��#�@aG�A��
C��                                    BxXSO�  �          A.�R?����p�@��RA�(�C�ff?����"{@s33A���C�\                                    BxXS^V  �          A/
=?�
=�ff@�
=A�{C��
?�
=�#
=@s�
A��HC�P�                                    BxXSl�  �          A.{?�{��@�33A�(�C�1�?�{� z�@~{A��C��)                                    BxXS{�  "          A.{?�ff�z�@���A�ffC��)?�ff�!�@y��A�  C���                                    BxXS�H  
�          A.�H?�ff�\)@���A�z�C�?�ff� ��@���A�z�C���                                    BxXS��  
�          A.�R?�  �
=@�Q�A���C��H?�  � Q�@��A��HC���                                    BxXS��  
�          A/33?���ff@��B��C���?���   @�=qA�C�@                                     BxXS�:  	�          A.�H?p���G�@�Q�Bp�C��3?p���33@�{A��C���                                    BxXS��  �          A.�H?xQ��{@�Bp�C�
=?xQ���@��A�  C���                                    BxXSӆ  
�          A.ff?��H��@�33B��C���?��H�33@�G�A�G�C�y�                                    BxXS�,  T          A-��?�=q�33@���A��HC�Z�?�=q�   @��\A���C�
                                    BxXS��  
�          A-�?�{�(�@�=qA���C�n?�{� ��@�  A���C�+�                                    BxXS�x  	�          A.{?�G��ff@���A�  C���?�G��!�@l��A�
=C�@                                     BxXT  �          A*�H?���  @�(�Bp�C�@ ?���G�@��A�(�C��                                    BxXT�  T          A(��?Y���G�@��B�C���?Y����
@�  A��C��                                     BxXT+j  D          A,  ?��\�ff@�B��C�\?��\��@�ffA�33C��R                                    BxXT:  2          A,��?�����@��B��C��H?������@�A�\)C�7
                                    BxXTH�  
�          A,��?h���  @��
B��C��H?h�����@�z�A�Q�C��                                    BxXTW\  T          A-G�?
=q�ff@���B
\)C��q?
=q�  @�{A�33C���                                    BxXTf  T          A-G�?333�	p�@љ�B�
C�U�?333�  @�z�A��C�q                                    BxXTt�  �          A-p�>\��@�p�B=qC�B�>\�@�  AۮC�%                                    BxXT�N  D          A-��>�{�
�\@У�B�HC��>�{���@��
A�G�C��                                    BxXT��             A)p����
��\@���B�
C�𤼣�
���@�ffA��C��3                                    BxXT��  S          A&{�u��(�@��B C�#׾u���@�(�A�C�:�                                    BxXT�@  
Z          A'�
��Q���(�@��B$��C�����Q��G�@�z�B�C��{                                    BxXT��  �          A'�>#�
�p�@�z�B�\C���>#�
��
@�33A�{C�~�                                    BxXŤ  
�          A'
=?�R���@ȣ�B��C�!H?�R�ff@��RA�
=C��\                                    BxXT�2  �          A'�
?�{��
@�{BffC��q?�{��@��\A���C�U�                                    BxXT��  �          A%p�?z��
=@���B��C���?z���\@�{A�p�C��                                     BxXT�~  "          A%?!G��z�@��HB	Q�C��?!G����@���A�33C���                                    BxXU$  "          A%p�?�����@���B
=C�h�?���p�@���Aۙ�C��                                    BxXU�  "          A#\)��G���@�B\)C�� ��G��33@�z�A��
C���                                    BxXU$p  "          A$��?fff�	�@�33B{C���?fff���@���A�{C���                                    BxXU3  "          A&{���	@���B�C��)�����@�\)A���C��                                    BxXUA�  �          A&�R����33@�\)B�C�}q�����R@�A�G�C���                                    BxXUPb  
�          A(�׽�����@��Bz�C���������@��AϮC��\                                    BxXU_  "          A+\)��\)��@��HB=qC�����\)�\)@���AǅC���                                    BxXUm�  "          A+������z�@�(�B�\C��=�������@��\A֏\C���                                    BxXU|T  �          A+\)>.{��\@�ffB�HC���>.{�ff@�z�AͅC���                                    BxXU��  "          A+33���
�R@�  B(�C������\)@��A�z�C��                                    BxXU��  "          A*�R>�  �(�@��
A���C���>�  �33@��\A�(�C��q                                    BxXU�F  �          A)��>����\)@�  B
=qC��>����33@�  A�p�C�                                      BxXU��  T          A(��>��R���@�B33C��>��R�ff@�\)A��C��R                                    BxXUŒ  �          A)p�>�p��33@��
B�HC�@ >�p���
@��A�C�%                                    BxXU�8  
�          A)p�>����	�@���B�C��q>�����@�A��C���                                    BxXU��  "          A+33?#�
�\)@��Bp�C��?#�
��R@��A�p�C��f                                    BxXU�  �          A)=L�����@�\)B  C�/\=L���p�@�=qA���C�+�                                    BxXV *  �          A'\)����{@�\B*(�C��׿����@�  B��C��                                    BxXV�  T          A&�H��33����@�G�B"��C��)��33��@�{BffC���                                    BxXVv  �          A'\)�#�
���@�z�B%G�C��=�#�
�
ff@��BQ�C���                                    BxXV,  "          A)�!G���{@��B-�C����!G��	�@ƸRB\)C��f                                    BxXV:�  �          A+���R��\)@���B.��C��3��R�	�@��HB
=C��\                                    BxXVIh  T          A)��(����R@�B,��C�c׾�(��	G�@�{B33C���                                    BxXVX  T          A&{>u��
=@���B'�C�� >u���@��B	�
C�˅                                    BxXVf�  T          A%p�=�\)� (�@�Q�Bp�C�C�=�\)�Q�@�{A��RC�=q                                    BxXVuZ  �          A%�u�   @�=qB�C���u�Q�@�Q�B ��C��=                                    BxXV�   T          A$�׾.{����@��B"ffC�]q�.{���@�z�B�C�l�                                    BxXV��  �          A$�׽#�
��@��
B/�C��ý#�
�  @�z�B�\C�ٚ                                    BxXV�L  T          A%<��陚@�\B4p�C��<��ff@��
B{C�)                                    BxXV��  "          A#
=�.{��  @��
B)��C�]q�.{���@���B��C�l�                                    BxXV��  T          A�R����Q�@��BC�
�����@��Aљ�C�&f                                    BxXV�>  �          A{�W
=��@��B ��C�<)�W
=��@�  A�{C�H�                                    BxXV��  �          AQ�8Q���@�\)A�{C�Y��8Q��
�\@~{A��HC�c�                                    BxXV�  T          A�þ�ff��
@��A��C�s3��ff�  @r�\A��C��=                                    BxXV�0  
�          A�R���
�
�H@�Q�A�
=C��\���
�{@L��A���C��q                                    BxXW�  �          Aff�����@��RB�HC�^����
ff@�
=A��
C�y�                                    BxXW|  "          A녿   �	�@�G�A�z�C�S3�   �(�@P��A��HC�g�                                    BxXW%"  
�          A�
�W
=�
�R@��A���C�<)�W
=��@U�A�C�^�                                    BxXW3�  T          A����R��@��HA�C�ٚ���R�33@e�A��C��                                    BxXWBn  �          A�������	@\)AĸRC�t{�����(�@>�RA���C���                                    BxXWQ  T          A녿��H�@\��A�(�C�S3���H�
=@�HAip�C�}q                                    BxXW_�  �          A�Ϳ���Q�@���A�z�C�⏿����R@C�
A�p�C�)                                    BxXWn`  
�          AG�����(�@�p�A�Q�C�Y������H@L(�A���C��\                                    BxXW}  
Z          A�׿�ff��@�z�A�p�C�����ff���@\(�A���C�Ф                                    BxXW��  �          Ap���G��33@�G�A�(�C�� ��G��
�R@w
=A�{C��q                                    BxXW�R  �          A�Ϳ��ff@���A��
C�녿��
{@xQ�A�{C�%                                    BxXW��  T          A녿�z�� ��@���A�G�C����z����@�z�A�{C�Q�                                    BxXW��  
�          A�R��{���@��B�C�)��{�
=@���A���C�j=                                    BxXW�D  �          A
=����@��
A��HC�����	�@�
=A�ffC�T{                                    BxXW��  �          AG���33�=q@�
=A޸RC�)��33���@c�
A�ffC�K�                                    BxXW�  "          Aff����
�\@�Q�Aģ�C��������@Dz�A��RC���                                    BxXW�6  �          A녿����(�@���A�{C�B������
=@j=qA���C�}q                                    BxXX �  
(          A
=���H� ��@�ffB�
C��Ὼ�H���@��HAԸRC�%                                    BxXX�  "          A
=�ٙ��33@��HA�Q�C�!H�ٙ��
�\@~{A��C�n                                    BxXX(  "          A�R������H@�Q�B
�RC��)����@�A�33C�K�                                    BxXX,�  "          A�R�������@�(�BQ�C�𤿰���Q�@��A���C�B�                                    BxXX;t  �          A
=���H�z�@�p�A�p�C�*=���H�\)@tz�A�(�C�q�                                    BxXXJ  �          Az��(����@�\)B
=C���(��	G�@�z�A�G�C�XR                                    BxXXX�  T          A���  ��H@��RA�\)C��
��  �
{@�(�A��HC�E                                    BxXXgf  �          A���(��p�@��
A�(�C��)��(����@��A�{C�N                                    BxXXv  �          Ap���\�33@�z�A�(�C�녿�\�
�\@�=qA�ffC�<)                                    BxXX��  "          A
=��ff���@��RA��RC��H��ff�  @�z�A�\)C�1�                                    BxXX�X  
�          A������@���A�
=C�J=�����@Q�A��C��H                                    BxXX��  "          A33��p���@�(�Aď\C��\��p���R@Q�A�{C��                                    BxXX��  "          A���(��ff@�G�A��C����(���
@K�A���C��q                                    BxXX�J  �          A\)�ff�G�@��
A�33C�P��ff��R@Q�A��C��                                    BxXX��  �          A   �33�@XQ�A�ffC޸�33�=q@!�AjffC�!H                                    BxXXܖ  �          A ����\���@n�RA�=qC�=��\���@8��A��C�q                                    BxXX�<  "          A!��p����@�
=A�p�C~z��p��33@Y��A�33C
=                                    BxXX��  "          A!�,(����@h��A�
=C}n�,(��G�@4z�A�\)C}�                                    BxXY�  T          A#�
�)����\@�p�A�\)C}p��)���  @W
=A�{C~�                                    BxXY.  "          A%G�������@�Q�A���C~W
����  @��A��
C�                                    BxXY%�  
�          A%�	���Q�@�  B (�C�{�	���\)@�\)A�G�C�B�                                    BxXY4z  �          A%p���
���@�\)B��C~ff��
�(�@��A���C0�                                    BxXYC   
(          A$���0  ��@�{B\)C{G��0  �	G�@�
=A�33C|5�                                    BxXYQ�  �          A$z��\)��@��\B
��C~�\�\)�	p�@��
A��C\)                                    BxXY`l  �          A%p���=q��@���B�
C�����=q�\)@��A��C���                                    BxXYo  "          A%p���{�  @��B�HC��f��{�\)@�ffA�Q�C���                                    BxXY}�  �          A%p���=q�p�@���B��C��\��=q���@��HA�z�C��                                    BxXY�^  
�          A&=q��ff���@�
=A�p�C�4{��ff�
=@��Ạ�C�s3                                    BxXY�  �          A'33�z��Q�@���A�z�C�Z��z���R@�G�A�=qC���                                    BxXY��  �          A'
=��(���@�
=BQ�C�t{��(���\@���A�RC�                                    BxXY�P  "          A&{��R��
=@�\)B"�C{����R��\)@ÅBC|��                                    BxXY��  
�          A%���	����\@�\)B33C5��	���	��@�=qA�p�C�3                                    BxXY՜  �          A%��(��
�H@�(�A�Q�C�c׿�(��G�@�{A�33C��                                     BxXY�B  T          A&�H�*�H��R@�
=A��HC}W
�*�H�(�@���A���C}�                                    BxXY��  
�          A'�
�5����@J=qA��HC}}q�5��Q�@�AV�\C}�
                                    BxXZ�  0          A&ff�Dz���\@ffAQG�C|^��Dz����?�\)A33C|��                                    BxXZ4  �          A&�\�7
=�(�@{AD��C}�f�7
=�=q?�p�A33C}޸                                    BxXZ�  
�          A&�R�a���R@,��Ap  CyxR�a����?��RA0Q�Cy�\                                    BxXZ-�  �          A(  �aG��ff@��AAG�Cy��aG��z�?��RA{Cz33                                    BxXZ<&  
�          A&ff�J�H�
=@33A5C{޸�J�H���?��@�p�C|{                                    BxXZJ�  "          A'\)�I���{@"�\A`��C{���I�����?�A"ffC|.                                    BxXZYr  �          A'�
�L���(�@�A;
=C{޸�L���=q?�
=@��\C|{                                    BxXZh  T          A'\)�mp��{?��HA�RCx��mp���?��
@�p�Cy)                                    BxXZv�  "          A&�\�hQ���\?�33@��RCyh��hQ���?:�H@\)Cy��                                    BxXZ�d  T          A*{�s33�=q?#�
@\(�Cx���s33��\�#�
�aG�Cy                                    BxXZ�
  �          A)��c33��H?�@FffCzW
�c33�33���Ϳ��Cz^�                                    BxXZ��  �          A$���c�
�=q?�\@7
=Cy�R�c�
��\���:�HCy�q                                    BxXZ�V  "          A%G��~{�Q�=�Q�?�Cwc��~{�(����>{Cw\)                                    BxXZ��  �          A%��o\)����k����\Cx���o\)���Y����  Cx�f                                    BxXZ΢  �          A$���g
=�=q=�Q�?�CyxR�g
=�{���;�Cyp�                                    BxXZ�H  T          A&{�j�H�\)�W
=��33CyL��j�H��H�Tz����Cy=q                                    BxXZ��  
�          A'\)�`���p��\��Czff�`����Ϳ�  ��\)CzQ�                                    BxXZ��  
�          A(Q��n{���
=�L��CyB��n{�Q쿙������Cy&f                                    BxX[	:  
�          A-���\��R��{���Ct����\�������;�Cth�                                    BxX[�  
�          A/
=����(��e��33Cv�H����z��������CvY�                                    BxX[&�  
�          A.�H�?\)��
�{����RC|��?\)��
��  ���HC|�                                     BxX[5,            A0z����
=q�����p�Ct&f����������
CsB�                                    BxX[C�  
�          A*�\�dz������=q�ͅCw��dz��Q���=q��=qCwE                                    BxX[Rx            A&�H�Dz��녿�(���\)C|�3�Dz��z����{C|��                                    BxX[a  "          A"=q� ���z�=�G�?�RC�H� ���Q��ff�!�C��                                    BxX[o�  T          A ���G��\)?��HA{C�"��G��Q�?h��@��C�,�                                    BxX[~j  	�          A)���{�   @HQ�A��C�!H��{�"�\@$z�Aa�C�/\                                    BxX[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX[�\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX[Ǩ  
          A)�&ff�((��E���ffC�<)�&ff�'\)�����z�C�9�                                    BxX[�N  
�          A-�;�� ���4z��qG�C}� �;��ff�Tz����HC}��                                    BxX[��  �          A.�H�4z��$���\)�=��C~��4z��#
=�0  �iC~�                                     BxX[�  T          A1��
=�+\)�.{�fffC�k��
=�+
=�333�j=qC�g�                                    BxX\@  
Z          A8  ?�(����@�
=B	z�C�9�?�(����@�Q�A�33C��                                    BxX\�  �          A7�
?У���
@ӅB�C��q?У����@��B�C��{                                    BxX\�  
�          A2=q?+��#�@�(�A�
=C��q?+��&�R@x��A���C��{                                    BxX\.2  T          A1��?B�\�!�@�{A��C�(�?B�\�$��@~{A�  C�q                                    BxX\<�  
�          A2{?0���$��@��RA�p�C���?0���'�
@o\)A���C���                                    BxX\K~  
�          A/�>���*=q@%AZ�\C�@ >���,  @
=A1��C�=q                                    BxX\Z$  
�          A2{>u�+�
@2�\Ai�C��H>u�-@z�A@��C��                                     BxX\h�  "          A9�?J=q�/�@j�HA�p�C��?J=q�2{@L(�A�C��                                    BxX\wp  "          A<��?!G��2�\@p��A��RC���?!G��4��@Q�A�\)C���                                    BxX\�  
�          A=�?!G��1p�@��A�(�C���?!G��4  @l(�A��C��)                                    BxX\��  x          A=�?L���1�@��RA�(�C��?L���3�@p  A��C�
=                                    BxX\�b  
�          A;�?���0��@tz�A�z�C�n?���333@XQ�A�{C�h�                                    BxX\�  �          A8z�?fff�.�\@fffA��C�]q?fff�0��@J�HA33C�W
                                    BxX\��  
�          A6�H?}p��*�\@~{A���C��?}p��,��@c�
A��C��)                                    BxX\�T  
�          A4z�?333�*=q@j�HA��C��?333�,Q�@P��A�z�C�޸                                    BxX\��  
�          A4Q�?5�*{@h��A���C��f?5�,(�@O\)A��C��H                                    BxX\�  �          A5�?�\)�,z�@Z=qA���C��
?�\)�.ff@@��AuC��\                                    BxX\�F  
L          A8z�?�(��.ff@S33A�
=C�|)?�(��0(�@9��AiC�p�                                    BxX]	�  0          A6�H?�33�1p�@p�A2�RC��)?�33�2�\?���A�RC���                                    BxX]�  �          A4��?�p��2�\?��@�G�C�%?�p��3
=?@  @u�C�"�                                    BxX]'8  T          A1G�?�=q�/�
?J=q@��HC���?�=q�0(�>�
=@
�HC�Ф                                    BxX]5�  "          A/33?У��,��?(��@\��C�K�?У��,��>���?�=qC�H�                                    BxX]D�  T          A1�@-p��+�?aG�@���C��@-p��,  ?�@/\)C��                                    BxX]S*  
�          A0��@9���*{?Y��@�z�C���@9���*ff?�\@)��C��                                     BxX]a�  "          A/�@\)�*�R?:�H@r�\C��\@\)�+
=>���@z�C���                                    BxX]pv  T          A0  ?����,��?0��@hQ�C�ٚ?����-�>�p�?�
=C��
                                    BxX]  
�          A.�\@p��)��?O\)@��C��f@p��)�?   @&ffC���                                    BxX]��  T          A/
=@�\�+�
>�@!G�C�` @�\�,  >.{?h��C�`                                     BxX]�h  �          A0��?��.ff>8Q�?s33C�XR?��.ff���#�
C�XR                                    BxX]�  T          A0  ?��H�->��?E�C�u�?��H�-����G�C�u�                                    BxX]��  T          A.=q?����+�>B�\?��\C���?����+���Q��C���                                    BxX]�Z  �          A-G�?�G��*�R>�ff@��C��?�G��*�H>8Q�?p��C���                                    BxX]�   "          A0��?޸R�.�R>u?�  C���?޸R�.�R���.{C���                                    BxX]�  
�          A.�R?�G��,�þ�=q��C��
?�G��,�Ϳ��3�
C��R                                    BxX]�L  
�          A-�?���,(��.{�c�
C��f?���,(���(��  C��                                    BxX^�  �          A+�
?��\�*�H��{����C���?��\�*�R�
=�H��C��)                                    BxX^�  
�          A+
=>��*ff�+��e�C�<)>��*{�fff���C�=q                                    BxX^ >  
�          A*ff>���)p��k���ffC��3>���)���33��ffC��3                                    BxX^.�  "          A+\)�k��*=q�����{C�aH�k��)녿��
��z�C�`                                     BxX^=�  T          A*{�333�)p�����N�RC�q�333�)��O\)����C�q                                    BxX^L0  �          A(�׿   �(  �
=q�=p�C���   �'�
�@  ���HC��                                    BxX^Z�  �          A)�>���ff@:�HA�G�C�l�>���\)@.�RAu�C�l�                                    BxX^i|  
�          A/�?aG���@�p�A��
C��3?aG����@��A�p�C���                                    BxX^x"  "          A2�\?�(����@�33B
�C�� ?�(���\@�{B�HC���                                    BxX^��  �          A3�?���  @�z�A�G�C�G�?���p�@�\)A��C�>�                                    BxX^�n  �          A4��?�\)���@��B�C�l�?�\)�ff@��B33C�c�                                    BxX^�  
�          A4Q�?���Q�@��B=qC�� ?���@��B��C�u�                                    BxX^��  
�          A4��?����(�@�(�B	�\C�N?������@ǮB�C�E                                    BxX^�`  �          A5�?����G�@�  B(�C�7
?�����R@��
B�HC�,�                                    BxX^�  �          A4z�?����=q@�
=B
=C��?�����@��HB�C��                                    BxX^ެ  �          A5�?��H�33@�
=B�C�� ?��H���@�33B�C��
                                    BxX^�R  S          A6{?�Q���@���B
=C�}q?�Q��33@���B33C�s3                                    BxX^��  
�          A6=q?�
=�(�@�{B�HC��?�
=�p�@ҏ\B=qC��)                                    BxX_
�  �          A6�\@�����@ۅB�
C��H@����@�Q�B\)C���                                   BxX_D              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX_'�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX_6�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX_E6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX_S�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX_b�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX_q(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX_�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX_�t              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX_�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX_��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX_�f              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX_�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX_ײ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX_�X              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX_��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`J              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX` �              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`/�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`><              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`L�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`j.              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`x�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`�z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`�               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`�l              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`и              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`�^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX`��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXaP              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXa�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXa(�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXa7B              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXaE�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXaT�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXac4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXaq�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXa��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXa�&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXa��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXa�r              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXa�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXaɾ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXa�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXa�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXa��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXbV              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXb�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXb!�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXb0H              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXb>�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXbM�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXb\:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXbj�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXby�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXb�,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXb��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXb�x              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXb�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXb��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXb�j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXb�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXb�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXb�\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXc              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXc�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXc)N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXc7�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXcF�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXcU@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXcc�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXcr�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXc�2  
�          ADQ�@33�5�@�  A�(�C��
@33�3�
@��HA��C��f                                    BxXc��  
�          AD��@!G��6=q@xQ�A��C�=q@!G��4(�@�\)A��C�N                                    BxXc�~  T          AD��@4z��4Q�@���A��C��@4z��2=q@�z�A�33C�R                                    BxXc�$  T          AE�@>�R�4(�@�Q�A�{C�j=@>�R�2{@��
A�=qC��                                     BxXc��  "          AD��@ ���6�H@s�
A�Q�C�0�@ ���4��@�A��HC�B�                                    BxXc�p  �          AD��@1G��6ff@o\)A�\)C���@1G��4Q�@��A�  C��                                    BxXc�  "          AE��@<(��6�H@hQ�A��HC�4{@<(��4��@�Q�A��C�G�                                    BxXc�  T          AEp�@?\)�6�H@c33A��C�W
@?\)�4��@{�A��\C�j=                                    BxXc�b  
�          AE�@G
=�733@VffA|z�C��R@G
=�5G�@o\)A�G�C���                                    BxXd  "          AE��@Y���5@X��A
=C�Q�@Y���3�
@q�A��\C�g�                                    BxXd�  �          AEp�@C33�6{@h��A���C��H@C33�3�
@�G�A���C��
                                    BxXd"T  
�          AEp�@K��6�\@\��A��C��=@K��4z�@vffA��C��                                     BxXd0�  "          AE��@HQ��7�@U�Ay�C�� @HQ��5��@n�RA��RC��{                                    BxXd?�  
�          AD��@8���733@[�A���C�3@8���5�@uA��C�&f                                    BxXdNF  �          AD��@'��7�@c33A�  C�l�@'��5p�@}p�A�=qC��                                     BxXd\�  �          AE�@(��8z�@`  A�=qC��R@(��6ff@z�HA��RC�
=                                    BxXdk�  �          AEG�@9���:�H@*=qAG�C���@9���9�@EAhz�C��                                    BxXdz8  T          AD��?�\�6�R@��
A���C�g�?�\�4(�@�G�A�C�w
                                    BxXd��  "          AD��?�(��6�\@�  A��C�G�?�(��3�
@�A��HC�XR                                    BxXd��  "          ADz�@��9p�@Tz�Az�RC��3@��7\)@p��A��\C���                                    BxXd�*  
�          ADz�@   �:�R@8Q�AY�C�
=@   �8��@U�A{�C��                                    BxXd��  "          ADz�@"�\�:{@@  Ab�RC�+�@"�\�8  @\��A��RC�<)                                    BxXd�v  "          AD��@#�
�9p�@N{As
=C�8R@#�
�7\)@k�A��C�J=                                    BxXd�  "          AEp�@"�\�:�\@FffAiG�C�'�@"�\�8z�@c�
A�ffC�8R                                    BxXd��  �          AD��@%��:{@=p�A_33C�@ @%��8(�@Z�HA�p�C�P�                                    BxXd�h  T          AC�@��:�R@0  APQ�C��@��8��@N{Atz�C��3                                    BxXd�  
�          AD  @(��;
=@8Q�AY�C�N@(��8��@VffA~�\C�\)                                    BxXe�  
�          AA�@��8z�@C33AiC�q@��6=q@aG�A�G�C�,�                                    BxXeZ  �          A<��?��
�4��@Mp�A|Q�C�<)?��
�2ff@j�HA���C�Ff                                    BxXe*   
�          A=G�?����4��@U�A���C��?����2=q@r�\A���C�3                                    BxXe8�  �          A<  ?333�2�H@^�RA�=qC��=?333�0z�@|��A�\)C�Ф                                    BxXeGL  �          A=�>\�4  @g
=A�=qC���>\�1G�@��\A���C��q                                    BxXeU�  T          A<��?
=q�4Q�@]p�A�=qC�]q?
=q�1@{�A�C�aH                                    BxXed�  
�          A:�\>�ff�333@J�HA|Q�C�'�>�ff�0��@h��A�C�+�                                    BxXes>  
Z          A:{?n{�2�R@C�
At(�C�aH?n{�0z�@b�\A��
C�h�                                    BxXe��  "          A9��?8Q��1p�@UA��C���?8Q��.�H@tz�A���C��H                                    BxXe��  	�          A9��?���.�H@p  A�G�C�n?���,(�@�\)A�\)C�t{                                    BxXe�0  �          A9G�?��
�3�@%AN�RC��q?��
�1p�@E�Aw
=C��                                    BxXe��  �          A:{?u�4(�@.�RAYG�C�p�?u�2{@N�RA���C�xR                                    BxXe�|  �          A:{?
=�3
=@EAv�HC�� ?
=�0��@eA��C��                                    BxXe�"  �          A9�?��2�R@8��Ah  C�T{?��0z�@X��A��\C�Y�                                    BxXe��  T          A;�?��
�2{@c33A�
=C���?��
�/\)@���A��C��                                    BxXe�n  
�          A;�?
=�3�@S33A���C���?
=�0��@s�
A��C���                                    BxXe�  �          A<Q�>����4��@N{A~ffC��H>����2ff@o\)A�(�C���                                    BxXf�  �          A9G�?!G��1��@N{A�
=C��H?!G��/
=@n�RA�=qC��f                                    BxXf`  �          A8Q�>�
=�2�R@2�\A`z�C�3>�
=�0z�@S�
A��C�
                                    BxXf#  "          A8z�>8Q��2�R@7
=Ae�C�u�>8Q��0Q�@XQ�A�ffC�xR                                    BxXf1�  T          A9��>���4z�@*=qATz�C���>���2ff@L(�A�C���                                    BxXf@R  "          A:=q?����4��@Q�A<��C���?����3
=@:=qAh  C��q                                    BxXfN�  T          A:�R?�G��2�R@:�HAh��C�8R?�G��0Q�@\��A�(�C�C�                                    BxXf]�  �          A<Q�����3�@X��A��\C��������0��@z�HA�z�C��q                                    BxXflD  T          A=p���
=�5�@QG�A�z�C��\��
=�333@tz�A�z�C��                                    BxXfz�  �          A<�ÿn{�333@h��A��C����n{�0(�@�p�A���C��R                                    BxXf��  	�          A<�׿:�H�0��@z�HA��C��:�H�-@�ffA��
C��                                    BxXf�6  �          A<Q�>����0z�@���A��C�
=>����-�@��\A�p�C�\                                    BxXf��  
�          A:�\?�R�-p�@��RA��C���?�R�)�@��A��C���                                    BxXf��  �          A:{?fff�,��@�
=A�Q�C�b�?fff�)�@��A���C�o\                                    BxXf�(  T          A:{?���-�@�=qA�{C��
?���*�\@��A��RC��                                     BxXf��  �          A:�\>�(��0(�@p��A��HC��>�(��,��@��A��C�#�                                    BxXf�t  	�          A<  >#�
�/�@���A��C�h�>#�
�,(�@�{A�ffC�k�                                    BxXf�  "          A<��?O\)�/�@�
=A��
C�)?O\)�,  @�Q�A���C�&f                                    BxXf��  T          A;�?�R�1��@n{A��C��R?�R�.ff@���A�
=C��                                     BxXgf  �          A9�>\)�0z�@h��A�(�C�` >\)�-G�@�ffA�\)C�aH                                    BxXg  "          A9>aG��1�@\��A���C��{>aG��.{@�Q�A��C��
                                    BxXg*�  
(          A8��?�(��2{@'
=AQG�C��f?�(��/�@J�HA�C��3                                    BxXg9X  
�          A8�׿^�R�(z�@qG�A�33C��׿^�R�%�@��A���C��R                                    BxXgG�  
Z          A8�׿��R�&�H@�  A��C��
���R�#
=@���A���C�w
                                    BxXgV�  "          A8z�����'
=@�z�A���C��
�����#33@�p�A��HC��
                                    BxXgeJ  �          A8zῆff�&{@��A�(�C�)��ff�!�@�33A��
C��                                    BxXgs�  �          A8��=��-p�@G
=A�{C�Q�=��*�\@j�HA�  C�S3                                    BxXg��  
�          A7�����4  @(�A0(�C������1�@1G�A`  C���                                    BxXg�<  
�          A7
=��33�2=q@�A0��C�
��33�0(�@0��A`��C�{                                    BxXg��  �          A733?fff�1G�@�RA5G�C�S3?fff�/33@3�
Aep�C�Z�                                    BxXg��  
�          A7
==�Q��0��@>{Ap��C�>�=�Q��-�@b�\A�z�C�@                                     BxXg�.  �          A7����
�1��@p�AG�C�W
���
�/\)@C33Aw�
C�N                                    BxXg��  
�          A8  �z��2�H@\)AHz�C��׿z��0��@E�Ay�C�~�                                    BxXg�z  "          A8���%��\)@���A���C}q�%���R@�p�A�z�C�                                    BxXg�   �          A9���"ff@��
A�z�C�,���@��A�z�C��                                    BxXg��  "          A:�\��z��*�R@��RA�Q�C�<)��z��&�\@���A̸RC�%                                    BxXhl  
�          A:�H��p��,��@�  A�
=C��H��p��(��@��\AÙ�C��                                    BxXh  "          A9���
�)�@�ffA�ffC�7
���
�$��@�Q�A���C��                                    BxXh#�  �          A;
=�����(��@���A�=qC�������$Q�@�33A���C��{                                    BxXh2^  
�          A<��>��R�7�
@\)AC33C��>��R�5p�@G
=Au�C�Ǯ                                    BxXhA  
�          A<z�@?\)�4  ?˅@�Q�C�p�@?\)�2ff@��A,Q�C���                                    BxXhO�            A<  @L���333?�
=@߮C���@L���1��@�\A   C��                                    BxXh^P  .          A<��@HQ��2{@33A4z�C���@HQ��/�@:=qAd��C��{                                    BxXhl�  "          A<(�@=p��1p�@!�AF�RC�xR@=p��/
=@HQ�Aw\)C���                                    BxXh{�  "          A<(�@8���1��@#�
AI��C�K�@8���/33@J�HAzffC�e                                    BxXh�B  �          A<(�@1G��2ff@ ��AE�C��)@1G��/�
@G�Av=qC�{                                    BxXh��  
�          A;�@/\)�2{@   AD��C���@/\)�/�@G
=Au�C�H                                    BxXh��  
�          A;�
@E�1@z�A"�HC���@E�/�@+�AS�
C�ٚ                                    BxXh�4  
�          A9G�@P  �/\)?�ffA33C�B�@P  �-p�@=qA@  C�XR                                    BxXh��  �          A9�@Vff�,��@A;
=C���@Vff�*=q@;�Ak�
C��)                                    BxXhӀ  
�          A6�H@�R�1��?�=q@�C���@�R�0(�?�Q�A�C��R                                    BxXh�&  `          A7�?�
=�2�\?��R@��C���?�
=�0��@�A*�HC��{                                    BxXh��  �          A<  ?�z��4(�@/\)AX(�C��?�z��1p�@W�A�\)C��                                     BxXh�r  �          A9?��0z�@G
=Ax��C���?��-p�@n{A��C��q                                    BxXi  �          A:=q?���/\)@tz�A���C��{?���+�@�A�\)C��)                                    BxXi�  �          A<  ?���9�(���N{C���?���:{    �#�
C�˅                                    BxXi+d  
�          A<Q�?�  �6{@-p�AUp�C���?�  �3\)@VffA��\C���                                    BxXi:
  
�          A<��?����5@�
A6�\C�O\?����3\)@<��Aj{C�Z�                                    BxXiH�  
�          A:�H?h���9�?�@'�C�<)?h���9�?�Q�@��C�>�                                    BxXiWV  T          A<��?5�8��@
=qA)�C�?5�6�\@3�
A\��C���                                    BxXie�  �          A;��#�
�4z�@G�Aw�C��R�#�
�1G�@p��A��
C��
                                    BxXit�  "          A:ff���4  @<(�AjffC��f���1�@e�A�33C��                                     BxXi�H  �          A7����,Q�@mp�A��C��׿��(��@��A�
=C���                                    BxXi��  �          A5녿���(Q�@��
A�{C��H����$Q�@�
=A�  C�k�                                    BxXi��  
�          A4  ��z��'
=@u�A�G�C�}q��z��#\)@�p�A��C�c�                                    BxXi�:  �          A3��@���  @�=qA��C|ff�@���
=@��A���C{ٚ                                    BxXi��  T          A2�H��{�33@�33A�Q�Ct����{�
=q@�33A�  Ct#�                                    BxXĭ  "          A2�H� ����
@���A�(�C0�� ����R@��A��C~��                                    BxXi�,  
�          A3
=��Q���@�G�Aי�Cv!H��Q��Q�@���A�Cuff                                    BxXi��  "          A3
=�C�
�@��\A�z�C|T{�C�
��@��
A�C{�\                                    BxXi�x  
Z          A2�H�\(���@�A�ffCy���\(��(�@��RA�
=Cy
                                    BxXj  
�          A3��|(��z�@���A�{Cvh��|(��\)@���A�(�Cu��                                    BxXj�  �          A5��������H@��HB�RCf��������
=@�  B��Ce33                                    BxXj$j  T          A4����ff��@�  A��Cc��ff��  @���B	��CbJ=                                    BxXj3  �          A5p���� ��@��HA�{Cgk�����G�@���Aԏ\Cfp�                                    BxXjA�  T          A6�\��33�陚@�Q�A��HCc޸��33��ff@�p�B�Cb}q                                    BxXjP\  T          A5��(����@�
=A�  Ci:���(����@�p�Bz�Ch                                      BxXj_  "          A5����������@��A�G�Ch�������=q@���A�  Cf�                                    BxXjm�  �          A4����
=��ff@��A��Cg�q��
=����@�Q�A�Cf��                                    BxXj|N  
�          A4  ��\)��ff@�z�A�ffCh(���\)��@�=qB�\Cf��                                    BxXj��  "          A2�H���H��Q�@�Q�A�z�Ci����H��@�ffB �RCg�{                                    BxXj��  T          A3\)��ff��=q@�  A���Cg�\��ff��
=@�p�B��Cf�                                     BxXj�@  �          A2�\��z���ff@���A��Ce�=��z���33@��B	�RCd�                                    BxXj��  "          A2�\��G���33@�=qB  Cg���G��߮@ϮBG�Cf\)                                    BxXjŌ  �          A2ff��
=��\)@�G�B{Cfp���
=���
@�{B33Cd�q                                    BxXj�2  I          A2{���\��G�@���BCgT{���\��@θRB  Ce�f                                    BxXj��  �          A1��\)���
@���BQ�Ch&f��\)��Q�@�{B�Cf                                    BxXj�~  "          A1�����
��p�@�G�B�Chٚ���
���@θRBffCgxR                                    BxXk $  �          A/����R��z�@���B
=Ci�����R����@�{B��Ch+�                                    BxXk�  �          A0����
=��
=@ƸRB	(�Ck���
=���H@�(�B�Ci��                                    BxXkp  �          A0�����\��p�@�\)B�Cic����\����@�z�Bp�Cg�)                                    BxXk,  "          A0Q���p���G�@�=qBp�Cf+���p�����@�ffBz�Cd�                                     BxXk:�  T          A/������=q@��BG�Cc�������{@���B�Ca�R                                    BxXkIb  T          A0Q����R��ff@��
B��Ce�
���R���@�  B�RCc޸                                    BxXkX  "          A/���Q���p�@�Q�Bp�Cd���Q�����@��
B"�CbJ=                                    BxXkf�  �          A+���z���33@޸RB"�Cd����z���{@陚B+�HCb��                                    BxXkuT  �          A,(���33��R@�G�B��Cl���33���H@�ffBz�Cj��                                    BxXk��  �          A1�������
=@�  B��ChǮ������=q@�z�B ��Cg#�                                    BxXk��  
�          A2�\������
=@׮B�Cg�H������=q@�(�B�HCe�R                                    BxXk�F  �          A1����=q��=q@�p�B�Ci����=q��{@ڏ\B
=Ch�                                    BxXk��  "          A0z���=q��  @�  B�RCl�=��=q��z�@�B��Ck�=                                    BxXk��  �          A*�\�������H@�p�B(�Cm�3������  @�33B(�Cl�                                    BxXk�8  T          A+�
��ff��  @�  B�\Cih���ff��(�@�z�B  Cg�                                    BxXk��  T          A)G������ff@��
B  Ch\)����Ӆ@�  BG�Cf�                                    BxXk�  
�          A&�H��Q���(�@���B=qCh����Q�����@�B�Cg.                                    BxXk�*  T          A(Q����H����@��BG�Ch�=���H��{@�z�B��Cgk�                                    BxXl�  T          A&=q���H���@�Q�A��Ch�����H��{@���B
G�Cgh�                                    BxXlv  T          A$Q������G�@�33B
=Cg�R����θR@�
=B=qCf�{                                    BxXl%  
�          A!������  @�
=B	�HCd���������@�=qB�Cb�                                    BxXl3�  T          A!���=q��  @�{B	
=Ca����=q��p�@���B=qC`@                                     BxXlBh  
�          A#
=��z���p�@�A�G�Cf�
��z���(�@�=qA�33Ce�H                                    BxXlQ  T          A!��������
@��HA�
=CeaH������33@�
=A�\Cd=q                                    BxXl_�  "          A"=q��=q��Q�@�p�A�p�Ch����=q��\)@���A��Cg�f                                    BxXlnZ  "          A�\�z��񙚾��R���C~���z���=�?\(�C~�f                                    BxXl}   
�          A=q�C33�\@�\B�(�CNxR�C33����@���B��RCG��                                    BxXl��  
�          A ���Vff�\@陚B�k�C:�=�Vff<�@��B�C3�=                                    BxXl�L  �          A{�Fff��\)@��B�=qC9��Fff>#�
@��B�\)C1�                                    BxXl��  
�          A��Tz��c33@޸RB\ffCb��Tz��I��@�z�Be�C_��                                    BxXl��  �          A�
�_\)����@��BK=qCeE�_\)�k�@أ�BT�RCb�                                    BxXl�>  �          AQ��J=q��z�@�ffB;(�Cm+��J=q��G�@θRBE��Ck0�                                    BxXl��  "          Aff��(���{@�z�B,z�Cz���(����
@�ffB8�Cy޸                                    BxXl�  	#          AQ�n{����@�Q�B�C���n{��\)@�33B+�C��\                                    BxXl�0  �          A����(�@��B�RC�����\@��RB,\)C��f                                    BxXm �              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXm|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXm"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXm,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXm;n              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXmJ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXmX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXmg`              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXmv              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXm��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXm�R              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXm��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXm��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXm�D              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXm��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXmܐ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXm�6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXm��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXn�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXn(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXn%�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXn4t              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXnC              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXnQ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXn`f              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXno              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXn}�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXn�X              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXn��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXn��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXn�J              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXn��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXnՖ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXn�<              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXn��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXo�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXo.              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXo�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXo-z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXo<               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXoJ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXoYl              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXoh              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXov�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXo�^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXo�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXo��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXo�P              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXo��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXoΜ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXo�B              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXo��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXo��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXp	4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXp�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXp&�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXp5&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXpC�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXpRr              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXpa   2          A���HQ��\)��Q���Cy��HQ��33>��?�p�Cy
                                    BxXpo�  �          A33��Q��
�R>�  ?�\)C����Q��
ff?�R@\)C���                                    BxXp~d  
�          A��/\)�G�@AK�
C|��/\)��
@p�Ap��C|�
                                    BxXp�
  "          A���$z��p�?��\@�p�C~5��$z��z�?�33A�C~)                                    BxXp��  
�          A��?\)�p�?���A<��C{Q��?\)�(�@z�A`��C{&f                                    BxXp�V  �          A��,(��\)@33AX(�C}�3�,(��@+�A|Q�C}��                                    BxXp��  
�          A  �8���p�@�(�A���Cz޸�8����\@�
=A�ffCz}q                                    BxXpǢ  "          A���1G��Q�@�HAfffC|�f�1G���\@1�A��HC|��                                    BxXp�H  T          A
=�'
=��?�A�RC~8R�'
=��\?��A*{C~�                                    BxXp��  
�          A���p���=�Q�?�C�%�p����>�ff@2�\C�"�                                    BxXp�  
Z          A(����H��?p��@�{C������H��H?�ff@���C��
                                    BxXq:            A�����?u@���C�Z���
=?���A   C�T{                                    BxXq�  ,          A��G��G�?8Q�@�C����G����?���@�33C��{                                    BxXq�  T          A�׿�z���s33���
C��R��z��=q����l(�C���                                    BxXq.,  T          A{���
����
�hQ�C��3���
�\)��(��Ep�C���                                    BxXq<�  
�          A�\?������p��,C�7
?��ᙚ�����$�C�#�                                    BxXqKx  
�          Ap�>����=q���C���>��z���Q��مC���                                    BxXqZ  �          Az���
=�)������C�>����z����mG�C�B�                                    BxXqh�  �          A��������p��,��C�b�����zῳ33��C�g�                                    BxXqwj  "          A녿��
�����ff�Q�C�⏿��
�=q�z�H��ffC��                                    BxXq�  
�          A{����=q�����!p�C�Uÿ���33���\� ��C�\)                                    BxXq��  T          Ap���G��G���  ��z�C�^���G��녿n{��(�C�e                                    BxXq�\  
Z          A�R�ٙ������
���C�� �ٙ���>�z�?�C��                                     BxXq�            Az��=q�@\)A�ffC�� ��=q� Q�@1�A�(�C�p�                                    BxXq��  �          A�H��33��(�?���AZ=qC�����33���@�Ayp�C���                                    BxXq�N  T          Aff�=p���@&ffA���Cz���=p��z�@8��A�p�CzO\                                    BxXq��  �          A�H�{�
�\@�A��C~��{�(�@�
=A�  C}�\                                    BxXq�  "          A�H�����?�p�A:ffC�����@�\AX(�C�
                                    BxXq�@  �          A�
��z���@5�A���C��3��z��	�@G
=A��C��H                                    BxXr	�  �          A!���У��
=@�ffB	(�C�aH�У��   @��RB�C�@                                     BxXr�  �          A!������@���B��C��\������H@�G�B��C��R                                    BxXr'2  �          A"�R�B�\�(�@��\A�{C����B�\�	p�@��A��RC�|)                                    BxXr5�  	�          A"�R������
@��A�
=C��������	�@��\A�p�C�q�                                    BxXrD~  �          A$  �O\)� (�@ʏ\B�C���O\)���@ҏ\B��C��                                    BxXrS$  �          A$Q쿜(���  @�=qB�
C��=��(���G�@��B&�
C�k�                                    BxXra�  �          A"�H��
=����@�p�BQ�C��=��
=��=q@���B#33C���                                    BxXrpp  T          A$  ��\�=q@�ffB(�C�� ��\��ff@�{B�HC��q                                    BxXr  �          A=q�p��Q�@�\)A��C}s3�p��=q@�\)A�{C}33                                    BxXr��  �          A��ff�\)@���A�p�C��3��ff��@�z�A�\C��R                                    BxXr�b  �          A���=q�ҏ\@�(�B5��C�@ ��=q��(�@��B<=qC��                                    BxXr�  �          A33��{�)��A(�B��RCw@ ��{�=qA	G�B��3Cu8R                                    BxXr��  �          Aff?xQ���AQ�B���C�G�?xQ쾏\)Az�B��HC��=                                    BxXr�T  �          Ap��E��   A  B��Cx�E��޸RA��B��
Cv+�                                    BxXr��  �          A�7
=���R@�=qA�33Cy���7
=��=q@���A��HCyT{                                    BxXr�  �          Aff�0��� ��@�33A�=qC{��0����{@��A��
Cz�)                                    BxXr�F  �          A��1���Q�@��B33Cy
=�1���@�  B�
Cx��                                    BxXs�  �          A
=�'���\)@��HB�Cz�3�'���33@�G�B�CzaH                                    BxXs�  �          A���Q����@��
B�C{��Q����@��B=qC{�
                                    BxXs 8  �          AG��+���(�@�p�B
z�Cz��+���  @��B�HCy�R                                    BxXs.�  �          A\)��
����@�G�A�{C}z���
��@�\)A��HC}:�                                    BxXs=�  �          A{�9�����\@��B6��Cs���9�����@ٙ�B;��Cr�f                                    BxXsL*  �          AG��z����@�
=A��C}n�z���ff@���A�ffC}5�                                    BxXsZ�  �          A���%��@��BffCzY��%���
@��\Bp�Cz
=                                    BxXsiv  �          A(��)����=q@�{B��Cx��)����@��HB$�
Cw�f                                    BxXsx  �          A\)�'
=��p�@ə�B-  Cw��'
=����@�{B1Cv�{                                    BxXs��  �          A(��$z����@�(�B.��CwY��$z�����@У�B3p�Cv޸                                    BxXs�h  �          A�������@�=qB:G�CwL������33@�{B>�
CvǮ                                    BxXs�  �          A\)�8���K�@��\Bt�HCc� �8���AG�@�z�Bxp�Cb8R                                    BxXs��  �          A
=�*�H��=q@��Bg=qCl�*�H�z=q@�=qBk(�Ck��                                    BxXs�Z  �          A
=�!G����\@�p�BcQ�Co���!G���@�  BgQ�Cn��                                    BxXs�   �          A�
�p����@�BY�Cu��p���Q�@��HB]=qCu�                                    BxXsަ  �          A�
�Q���ff@ۅB@��CmǮ�Q����@޸RBDQ�Cm�                                    BxXs�L  �          Aff�`�����@�p�B4�Cm���`����\)@أ�B8�\Cmc�                                    BxXs��  �          A(��B�\����@�=qB${Ct.�B�\��p�@�B'�HCs�                                    BxXt
�  �          A���*�H����@���B*z�Cv��*�H���@�z�B.G�Cv��                                    BxXt>  �          A=q�U��@�{B-�Cp�f�U��=q@�G�B133Cp+�                                    BxXt'�  �          A�R�:�H��Q�@�33B:�HCs)�:�H��z�@�ffB>\)Cr�H                                    BxXt6�  �          A
=�U��
=@�
=B5�Co���U���@��B8��Co8R                                    BxXtE0  �          AG��#�
��G�@��HB,�Cw���#�
��@�{B/�Cw}q                                    BxXtS�  T          A33��G���(�@�p�B)��C�����G�����@ȣ�B-�C��f                                    BxXtb|  �          A���\��p�@��BL�CyY���\���@�BO\)Cx�R                                    BxXtq"  �          A33�=q���@��BW(�Cs���=q��  @�\)BZ(�Cs&f                                    BxXt�  �          A
=�E��  @�Q�B\{Ci��E��z�@�=qB^�Ci@                                     BxXt�n  �          AQ��^�R�K�@��RBmz�C^k��^�R�Dz�A (�Bo�\C]n                                    BxXt�  �          A33�@���]p�A\)BsG�Cd�R�@���VffA(�Bu�Cd�                                    BxXt��  T          AQ��K��7
=A�B|�C]�f�K��0  AQ�B~�
C\�\                                    BxXt�`  T          A���c33�\)A�B{��CW��c33�Q�A(�B}=qCU޸                                    BxXt�  �          A
=�C33�>{AffB|\)C`B��C33�8Q�A�HB~=qC_L�                                    BxXt׬  �          A���!G��n�RA33Bu  Cl  �!G��h��A�
Bw�CkT{                                    BxXt�R  �          A�\�8���X��AQ�Bv��Ce�\�8���S33A��Bx��Cd�\                                    BxXt��  �          A��HQ��<(�A(�Bz(�C_:��HQ��7
=A��B{�RC^c�                                    BxXu�  �          A��<���B�\A��B{�Caٚ�<���=p�AG�B}z�Ca{                                    BxXuD  �          A��{�^�RA��B~�HCmu��{�Y��A=qB�Q�Cl޸                                    BxXu �  �          A���	���A�AQ�B��HCj���	���=p�A��B��Cj                                      BxXu/�  �          AQ��ff�%�A	G�B��qCc�q�ff� ��A	��B�k�Cb��                                    BxXu>6  �          A��
�H�S�
AQ�B��HCl���
�H�O\)A��B���Cl.                                    BxXuL�  �          Az���
��
=@�\)Bmz�C{���
���A (�Bo
=Czٚ                                    BxXu[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXuj(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXux�   m          Az��333�g�A
=qBxffChB��333�dz�A
�\ByffCg�H                                   BxXu�t  �          A  �޸R��\)A	p�Bw�HCw���޸R��A	Bx�Cwh�                                   BxXu�  �          A  �(���33@�BK��C�!H�(���=q@�\BL�C�q                                   BxXu��  �          AQ�k����H@�p�BEC��þk����@�{BF�C��{                                   BxXu�f  �          A�׿������A��Bi�Cx��������A��Bi�
Cxff                                    BxXu�  �          AQ��!���ffA�HBoQ�Cn���!���p�A
=Bo�HCn�q                                    BxXuв  �          A\)�ff��33Ap�BmQ�Cq�{�ff���\A��Bm�
Cqu�                                    BxXu�X  �          Az���R���A�HBo{Cv����R��\)A
=Boz�Cv�                                    BxXu��  �          AQ���H��
=@�{BPffCv&f��H���R@�ffBP�RCv)                                    BxXu��  �          A��\�����
@��HBC�HCm:��\�����@��HBD
=Cm5�                                    BxXvJ  �          A�R�0  ���@�  BU��Cq}q�0  ����@�  BU�Cqz�                                    BxXv�  �          A���1G���A z�BcCm�3�1G���A z�Bc�Cm��                                    BxXv(�  �          A��!G��s33AffBs(�Clu��!G��s33A=qBs  Cl��                                    BxXv7<  T          A{���S33A\)B�� Cm����S�
A33B�aHCm�                                    BxXvE�  T          AQ쿧���Q�@�RBNz�C��q�������@�ffBN
=C��H                                    BxXvT�  �          A
=�L������@��B>  C��\�L����p�@�G�B=p�C���                                    BxXvc.  �          A!G���  ��z�@�B@p�C��ÿ�  ���@�
=B?C���                                    BxXvq�  �          A zῦff��G�@�(�BO�\C��Ϳ�ff��=q@��BNC��{                                    BxXv�z  �          A Q���
��  @��BT{C|�{���
����@�
=BS33C|�                                    BxXv�   �          A
=<���(�@��Bz�C�  <����@���BffC�                                      BxXv��  �          A�R�=p��ָR@�ffB;ffC��)�=p���  @�p�B:33C��H                                    BxXv�l  �          A���5��=q@���BC�N�5��@�33Bp�C�P�                                    BxXv�  �          A�Ϳ   ��{@��
B  C�33�   ��\)@�=qB�\C�4{                                    BxXvɸ  �          A�\@&ff�33@��Ak�
C�)@&ff��@��AeC��                                    BxXv�^  �          Aff@&ff�{?�z�A
=C��3@&ff�=q?˅Az�C��                                    BxXv�  
�          Az�?�ff�(�@L(�A�G�C��f?�ff�z�@G�A���C���                                    BxXv��  
�          AQ�˅��p�@�  B=qC�(��˅���R@�B G�C�0�                                    BxXwP  �          A�׿Ǯ�J�HA�HB��fCs�R�Ǯ�P  AffB���CtQ�                                    BxXw�  �          Aff��{=�Q�A\)B�B�C1W
��{�#�
A\)B�G�C4B�                                    BxXw!�  �          A!G�������33A{B�p�C`+������   AB���Ca�H                                    BxXw0B  �          A#
=��\����A
�\Bj{Ctc���\���
A	��Bg�
Ct��                                    BxXw>�  �          A#�������AQ�BW�RCx�\�����ffA33BU=qCx��                                    BxXwM�  �          A#��Q���Q�A�BQ��Cw���Q����A ��BOffCw��                                    BxXw\4  �          A"�\�����  @���BH  C}(������33@�ffBE=qC}h�                                    BxXwj�  �          A"�H��G����@���B;z�C���G����@��B8z�C��{                                    BxXwy�  �          A"ff�����ff@�{B%G�C�,Ϳ����G�@ҏ\B"33C�>�                                    BxXw�&  �          A z��ff��@�ffB(�C���ff��\@��HB%G�C��                                    BxXw��  �          A!����
=�G�@�z�B�RC�����
=��\@���BQ�C��                                    BxXw�r  �          A#33�����
=@�ffB
=C�������z�@�=qBz�C��                                    BxXw�  �          A#
=��  �Q�@��B33C����  �@���B�\C�3                                    BxXw¾  �          A#33�\)� ��@�  B	p�C|Ǯ�\)�{@��
BC|�R                                    BxXw�d  �          A#��1G���p�@�z�B  Cz!H�1G�����@�Q�B=qCzaH                                    BxXw�
  �          A$(��1G�����@��HB�Cz\)�1G���(�@�ffB��Cz�)                                    BxXw�  �          A#��!���z�@��B�C|.�!��   @��HBz�C|h�                                    BxXw�V  �          A#33�(����{@ÅB�C{��(�����@��RB�C{O\                                    BxXx�  �          A"�H� ����(�@��
B�HC��� ���   @�
=Bz�C�)                                    BxXx�  �          A#\)��(����@�{B�C�  ��(���@���B
{C��                                    BxXx)H  �          A#\)�   ���H@�ffB�
C���   ��
=@�G�B(�C��                                    BxXx7�  �          A"�R�\)��\)@�p�B�C{�
�\)���
@�Q�B  C{�f                                    BxXxF�  �          A#�
��R��@�Q�B33C|���R��=q@��HB\)C|h�                                    BxXxU:  �          A#�
�1���Q�@�(�B�Cy�R�1����@�ffB33Cz�                                    BxXxc�  �          A#��)����z�@���B{CzJ=�)����G�@˅B  Cz��                                    BxXxr�  �          A$  �&ff��\)@�Q�B�RCzٚ�&ff��(�@ʏ\B�C{8R                                    BxXx�,  �          A$������33@ۅB&��C|Ǯ�����@�p�B!ffC}#�                                    BxXx��  �          A%���
��@���B$\)C~����
���@ҏ\BC~�                                    BxXx�x  �          A#������
@߮B-{C}u����陚@��B'\)C}�
                                    BxXx�  �          A#��-p��׮@�B2�Cx��-p���@߮B,�HCx��                                    BxXx��  �          A"�R�6ff��{@��HB8Cv��6ff��z�@��B3
=Cv                                    BxXx�j  �          A#
=�W����@�\B@ffCpJ=�W����
@��B:�
Cq#�                                    BxXx�  "          A$  �I���Å@�33B?Q�Cr�R�I���ʏ\@�p�B9�Cs��                                    BxXx�  �          A$�Ϳ�z���\@�(�B(�Cc׿�z�����@��B"p�C�                                     BxXx�\  �          A&=q�ff��  @��HB%33C~\)�ff��{@ӅB��C~�                                     BxXy  �          A'����H���@��BF�ChaH���H����@�  BA\)Ci�{                                    BxXy�  T          A'���G��~�RA	�B[ffC]:���G����A
=BV��C_\                                    BxXy"N  
�          A(z����R��z�@�
=BEQ�Ch����R��z�@�G�B?�\CiB�                                    BxXy0�  T          A(���   ��(�@޸RB'  C{Q��   ��33@ָRB   C{��                                    BxXy?�  �          A)p���
=�	@�B	=qC�R��
=���@�z�B��C�,�                                    BxXyN@  �          A!G��W���  @��B:G�Cp���W��Ǯ@��B3�\Cq�)                                    BxXy\�  �          A Q�L����@�G�A��C��R�L���
=q@�  A�=qC�ٚ                                    BxXyk�  �          A!��>.{���\@�Q�BffC��q>.{� ��@�\)Bp�C���                                    BxXyz2  �          A!p��\����(�AQ�B`
=Cg�q�\����p�ABY�\Ci��                                    BxXy��  �          A!����Q�����A�BZ{Cb޸��Q���=qA�BS��Cd                                    BxXy�~  �          A$Q���=q�w�A�BX��C\G���=q��A�RBSffC^k�                                    BxXy�$  �          A$����33�@��A	p�Ba�CS�{��33�U�A�B\�
CVxR                                    BxXy��  �          A%p�����>�RA	Ba(�CS0�����S33A�
B\z�CU��                                    BxXy�p  �          A%����{�*=qAp�Bj\)CQ���{�@  A�Be��CT�\                                    BxXy�  �          A%p���  �(�A�\Bn\)CL�q��  �"�\A�Bj\)CP)                                    BxXy�  T          A&{��  ��\)A(�Bq��CIxR��  ��RA�HBn\)CM!H                                    BxXy�b  T          A$z������<��Ap�Bp(�CWk������S�
A\)Bj�\CZ�\                                    BxXy�  �          A$Q��Fff��  A�
BU��Cop��Fff���HA Q�BMCp�R                                    BxXz�  �          A%G��J�H��
=A�\BQ  Coٚ�J�H����@�BH��CqQ�                                    BxXzT  �          A%�,����AffBX�RCs���,������A�\BO��Ct��                                    BxXz)�  �          A%����{A�BXz�Cy�������A�BO�C{
=                                    BxXz8�  �          A%��\���\A�HBY�Cw����\��{A�HBPQ�Cx�                                    BxXzGF  �          A%p��8�����\A z�BK�Cs���8����@�Q�BB�Ct�                                    BxXzU�  �          A&{�Fff��\)A�\BXz�CoW
�Fff���A�\BOz�Cq�                                    BxXzd�  �          A%p��S�
���A	�Bb�CjL��S�
��  AffBY�HClxR                                    BxXzs8  �          A#�
�C�
��  A�BV(�Co��C�
��(�@�\)BL�
CqxR                                    BxXz��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXz��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXz�*   �          A((��~{�UA=qBqC\{�~{�q�A�BjQ�C_��                                   BxXz��  T          A'�
�l���vffA(�BmQ�Cb(��l����G�A��Be  Ce:�                                   BxXz�v  �          A(���*=q��Q�A�
B_�Cs&f�*=q��{A\)BU��Ct��                                   BxXz�  �          A)��
=����A
�RB]=qCv�
�
=��ffA{BR�Cx\)                                   BxXz��  �          A)��R����A  BL=qCzu���R��ff@�p�BA(�C{�{                                    BxXz�h  �          A*�\�'
=���A�RBHz�Cw��'
=��\)@��\B=p�CxǮ                                    BxXz�  �          A+
=�#�
���
@���B:��Cy�\�#�
���@��B/z�Cz��                                    BxX{�  �          A+
=�"�\��\@�\B5  Cz:��"�\��
=@�ffB)z�C{33                                    BxX{Z  �          A,(��%����A ��BA��CxǮ�%���\@�{B6=qCy�3                                    BxX{#   �          A,Q��-p���=q@��
B<  Cx^��-p���@�B0Q�Cy�                                    BxX{1�  �          A,z��,���ۅ@��
B;�Cx���,������@�\)B/�Cy��                                    BxX{@L  �          A,(��5��33@���B9�Cw� �5���@�z�B-Cx�3                                    BxX{N�  T          A,���8����\)@��RB6G�Cwz��8������@陚B*33Cx�f                                    BxX{]�            A,���3�
��@�B3ffCxxR�3�
��G�@�{B'�Cy�{                                    BxX{l>  �          A-G��0  ��=q@�
=B.Cyc��0  ���@���B"=qCzn                                    BxX{z�  �          A.�\�G
=��=q@�p�B:Q�Cu� �G
=���@�Q�B.  Cv�)                                    BxX{��  �          A-��Dz���
=@�\)B<�
CuxR�Dz���@�=qB0\)Cv޸                                    BxX{�0  �          A,���>�R��
=@�B5�Cvٚ�>�R��p�@�  B(Cx�                                    BxX{��  �          A,���<����@�\)B7  Cv��<����(�@陚B*�Cx5�                                    BxX{�|  �          A-��>{��@�\)B.Cw�R�>{��{@��B!�Cx�f                                    BxX{�"  �          A-��8����
=@�Q�B0  Cx8R�8����@ᙚB"�RCyc�                                    BxX{��  �          A.ff�;����H@�  B5G�Cw�\�;����@�G�B'�HCx�
                                    BxX{�n  �          A.�\�8Q��ۅ@�
=B;��Cw8R�8Q���@��B.Q�Cx��                                    BxX{�  �          A-�Fff��A�HBC��Ct8R�Fff��{@�  B6=qCu�f                                    BxX{��  �          A-��G���G�AffBL(�Cr���G���=qA   B>�
Ct�
                                    BxX|`  �          A,z��7
=����A\)BO{Ct���7
=���A ��BAffCvn                                    BxX|  
�          A+�� �����@�\B3��Cz�f� ������@�\B%ffC{��                                    BxX|*�  �          A*�\�����@�G�B4ffC|Y�������@ᙚB%�RC}h�                                    BxX|9R  �          A)�������
=@�
=B%(�C�C׿�����H@�{B{C��                                     BxX|G�  �          A)���޸R����@޸RB$�\C��\�޸R��
@�p�B=qC�f                                    BxX|V�  �          A)�������@ȣ�B  C�Uÿ���=q@�p�BG�C��                                     BxX|eD  �          A)�����@�{BffC�T{����	�@ÅB��C���                                    BxX|s�  �          A)��������@�\B'��C��H�����z�@�Q�B��C��H                                    BxX|��  �          A*{�����@�z�B0p�C������ ��@ڏ\B G�C�O\                                    BxX|�6  �          A)p��k���@��
B0p�C���k�� ��@��B {C��q                                    BxX|��  �          A(zᾔz���(�@��HBBC����z���@�=qB2(�C��                                    BxX|��  �          A+�
�u�޸RA   BC�
C���u����@�
=B3{C��                                    BxX|�(  �          A.ff�&ff��{A=qBA��C�lͿ&ff����@��HB0�C���                                    BxX|��  �          A.�H�B�\�
=q@���B�
C�~��B�\�@�\)B��C��                                     BxX|�t  �          A.ff�L����@�Q�Bz�C�aH�L����H@��\BG�C���                                    BxX|�  �          A.�\�z�H�
=@У�B�
C��=�z�H��\@��HB�C��{                                    BxX|��  �          A.=q�B�\��H@�B
G�C���B�\�{@�
=A�p�C��\                                    BxX}f  �          A-G��!G���@�B�C��)�!G��G�@�
=A�RC�{                                    BxX}  �          A,zᾣ�
��@�G�BG�C������
��
@�=qA�RC�                                    BxX}#�  �          A,(���33��R@�  B��C��׾�33�@���A�
=C���                                    BxX}2X  �          A,  ����G�@��
B
�
C��H������@�z�A��C��f                                    BxX}@�  �          A+�
>aG��(�@��HB�C���>aG��33@��HA�\C���                                    BxX}O�  �          A,Q�=��
�\)@��B��C�E=��
��@�ffB=qC�AH                                    BxX}^J  �          A,�׽�\)��
@���B\)C�� ��\)�(�@�{B��C���                                    BxX}l�  �          A-��u�G�@�Q�B\)C��\�u���@�Q�A��HC���                                    BxX}{�  �          A-G�>�(���@��B��C�Y�>�(���@�G�A��
C�H�                                    BxX}�<  �          A-p�>�Q���
@���B=qC�#�>�Q���@�  A�  C�{                                    BxX}��  �          A,(�>��H�  @�\)B��C��)>��H�  @�
=A�\C��f                                    BxX}��  �          A-�?k��@�(�B�C�&f?k���R@�z�B	�C��{                                    BxX}�.  �          A-G�>aG���R@��B {C��>aG��{@�AظRC��f                                    BxX}��  �          A-�=����(�@��\A��RC�K�=����\)@�Q�A��HC�G�                                    BxX}�z  �          A,�þ.{�\)@��A��RC�z�.{��\@��HAԣ�C��H                                    BxX}�   �          A-��>aG��  @�p�A�  C��\>aG��\)@��\AӅC���                                    BxX}��  �          A-G�>��33@��RA��RC�ff>���R@��
A��C�aH                                    BxX}�l  �          A-�?!G���@��
A�(�C��
?!G��
=@���A�33C�޸                                    BxX~  �          A,��?��ff@��RA�G�C���?��{@��A��C��\                                    BxX~�  �          A,��>�33��\@�
=A��C�
>�33�=q@��A��C��                                    BxX~+^  �          A-p�>�(���@��\BG�C�W
>�(���@�
=A�z�C�Ff                                    BxX~:  �          A,��    �  @�{B��C�H    �(�@��\A���C�H                                    BxX~H�  �          A,�þ���@�(�BC��R����@�Q�A���C���                                    BxX~WP  �          A-�>Ǯ��@�\)A��\C�5�>Ǯ��@��HAӅC�&f                                    BxX~e�  �          A.{>.{��\@�33BG�C��f>.{��R@��RA�33C�~�                                    BxX~t�  �          A->�G��@�Q�B�C�o\>�G���R@�z�A�{C�Y�                                    BxX~�B  �          A-p�?(��  @���B�C��3?(��z�@�(�A��
C��
                                    BxX~��  �          A-?(��  @�G�B33C��3?(����@�z�A�{C��
                                    BxX~��  �          A->�33�  @���B��C�&f>�33�G�@���A�z�C�{                                    BxX~�4  �          A-���	p�@�z�BC��\���33@�Q�B 33C��
                                    BxX~��  �          A.=q��\)�
{@��
B��C��H��\)�  @��A�z�C��f                                    BxX~̀  �          A-p��L����@�B�C�XR�L�����@���A���C�b�                                    BxX~�&  �          A,�ÿ!G��	�@�G�B
=C��=�!G��
=@���A�{C��                                    BxX~��  �          A-�c�
�	��@љ�B�\C�Ϳc�
��@���A���C�@                                     BxX~�r  �          A.�\����\@�p�Bz�C�����@���B33C�5�                                    BxX  �          A.=q�k���@��HB�\C�8R�k��{@�B
=C�Ff                                    BxX�  �          A-���Q����@�33B�C��
��Q��33@�A��C��)                                    BxX$d  �          A-������	�@��
B��C��H�����@�{A��C��=                                    BxX3
  �          A.{�����@��B=qC�ᾅ��{@�z�B=qC�+�                                    BxXA�  �          A.{>�p��Q�@�Q�B�RC�:�>�p���H@��\B�\C�#�                                    BxXPV  �          A.{?(��
=q@ҏ\B�C��?(����@�(�A�33C��                                    BxX^�  �          A-�>�\)�
=@�z�B$33C���>�\)��\@�
=B�C��=                                    BxXm�  �          A-�>�G��@�RB&Q�C���>�G��p�@�G�B��C�k�                                    BxX|H  �          A.ff>\)�
�\@�33B(�C�xR>\)��@��
A���C�o\                                    BxX��  �          A-�>.{�z�@��BG�C���>.{�{@�
=A�z�C�~�                                    BxX��  �          A.ff�:�H��R@�z�B	��C���:�H���@�(�A�\)C���                                    BxX�:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXņ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�,  )          A.�\>�\)�
=q@�(�B  C��\>�\)�G�@��
A�Q�C��q                                   BxX��  �          A-���������@޸RB��C�ٚ�����Q�@�
=B��C���                                   BxX�x  �          A.{������p�@�33B1��C��3������@�B�
C�8R                                   BxX�   �          A.ff�����@��B:��C}}q��� Q�@�G�B!=qC.                                    BxX��  �          A.�\�
�H��G�@�=qB7G�C}h��
�H�=q@�BC�                                    BxX�j  �          A.�H�	������@�Q�B-�C~E�	����@�=qB�RC��                                    BxX�,  �          A/33�33��z�@�
=B%=qC}�R�33�
�H@�  B\)C!H                                    BxX�:�  �          A/�
��{���@�\B �\C�Y���{���@��B�
C���                                    BxX�I\  �          A/��k��
ff@�  B��C�AH�k��=q@�A�G�C�P�                                    BxX�X  �          A/�>��
�G�@���Bz�C��>��
��
@���A�=qC���                                    BxX�f�  �          A/�>L����@��
B�RC���>L���\)@���A�ffC���                                    BxX�uN  �          A0�׾�Q��\)@�(�A��C�녾�Q��!�@�ffA��\C��)                                    BxX���  �          A/���G��=q@�Q�A���C�����G���
@��\A�\)C��q                                    BxX���  
�          A/�
?G��
=@���BG�C���?G��
=@�G�A�RC�^�                                    BxX��@  �          A0  ?+��Q�@�(�BG�C�?+���R@�ffA�Q�C��                                    BxX���  �          A0��?�ff�Q�@�Q�A�\)C�'�?�ff�"{@�G�A���C��
                                    BxX���  �          A0��?^�R���@�Q�A��C���?^�R�%��@p  A�
=C�h�                                    BxX��2  �          A/�?���z�@�ffA֣�C�?���%�@l(�A�\)C���                                    BxX���  �          A/33?@  �{@�A�=qC�.?@  �&{@Y��A��RC�3                                    BxX��~  �          A.=q?u��H@�(�A�G�C��
?u�#�@g�A��C���                                    BxX��$  �          A.�R?\(��  @��\A�=qC��H?\(��$��@c�
A�{C�aH                                    BxX��  �          A/���G��ff@���A�  C����G�� ��@���A�G�C��3                                    BxX�p  �          A/�=���
@�  A��\C�Z�=��!�@�\)A��C�U�                                    BxX�%  �          A/���\)�@��A�G�C��=��\)�#�@���A��
C���                                    BxX�3�  �          A0  ���
�z�@���Aٙ�C�3���
�%��@mp�A��C�                                      BxX�Bb  �          A0  �u���@�Q�A��HC��ýu�%@l(�A��HC��R                                    BxX�Q  
�          A0  ���
�Q�@�G�A�Q�C�����
�%��@mp�A��C���                                    BxX�_�  �          A/���Q��  @���A�{C��)��Q��%�@l(�A�\)C��                                     BxX�nT  �          A/33��=q�z�@��A�G�C�8R��=q�%��@dz�A�=qC�B�                                    BxX�|�  �          A.ff�.{��@�A��C��H�.{�$��@eA�C���                                    BxX���  �          A.=q�#�
�@��A�(�C���#�
�#\)@q�A�z�C��)                                    BxX��F  �          A/33>L���(�@��RAי�C���>L���%p�@fffA���C���                                    BxX���  �          A0  >������@��
A�ffC�'�>����&�R@_\)A�{C�
                                    BxX���  �          A.{?Tz��ff@�A�{C�c�?Tz��&�R@A�A��C�Ff                                    BxX��8  �          A/�?��
�&{@<(�AyC��)?��
�+
=?��R@��HC��R                                    BxX���  �          A0��@'��*=q?��H@�33C���@'��+�
���
����C��)                                    BxX��  �          A1�@5��*ff?}p�@��C�o\@5��+
=�   �%C�j=                                    BxX��*  �          A0��@%��*ff?�{@��HC�Ф@%��+����333C�                                    BxX� �  �          A1�@333�)p�?��A   C�e@333�+33=#�
>aG�C�S3                                    BxX�v  �          A0��@���)?�A�C�~�@���,Q�>���?�Q�C�g�                                    BxX�  �          A0��@��*�\?У�A�C�
=@��,z�=���?�C���                                    BxX�,�  �          A0��?��)�@
�HA6{C�)?��-�?#�
@U�C�                                    BxX�;h  �          A0��?�ff�)�@!G�AS33C�%?�ff�-�?z�H@�=qC��                                    BxX�J  �          A2{?xQ��'
=@n�RA��\C���?xQ��-@(�A5��C���                                    BxX�X�  �          A1G�?aG��%p�@xQ�A�  C�o\?aG��,��@ffAC�C�T{                                    BxX�gZ  �          A0��?fff�#�@�(�A�G�C���?fff�+\)@&ffAY��C�e                                    BxX�v   T          A0��?0��� z�@���A���C���?0���)p�@B�\A�
C���                                    BxX���  �          A1>���� ��@�A�C��q>����*{@J=qA�z�C���                                    BxX��L  �          A1�>L����
@�
=A�ffC���>L���)G�@L��A��RC���                                    BxX���  �          A0��=L���\)@�33B�C�'�=L��� (�@�p�A�=qC�%                                    BxX���  �          A0  =�\)�
=@�  BQ�C�<)=�\)�p�@�(�A�Q�C�7
                                    BxX��>  �          A/33�����@θRB  C��
����R@�=qA�\)C��                                     BxX���  
�          A.ff����
�H@ҏ\B��C�� �����@�ffA�(�C���                                    BxX�܊  �          A.{��\)��H@��
B�RC�
=��\)��@���A�(�C�"�                                    BxX��0  �          A-녾�=q���@�{B
=C�)��=q��@�=qA�z�C�1�                                    BxX���  �          A.�H��ff�  @�B	�RC��þ�ff�p�@��A͙�C��{                                    BxX�|  �          A0(��#�
�  @���A�\C��f�#�
�#�@�G�A�=qC��\                                    BxX�"  �          A/\)������@�ffBz�C�xR�����@��A���C��\                                    BxX�%�  �          A.�H��Q����@޸RB�C��ÿ�Q���@��HA��
C�`                                     BxX�4n  �          A.�R��{��\@�33B#{C�%��{��R@�  A�p�C��\                                    BxX�C  �          A+\)����G�A��BE  Cy������
=@�\)B#�C|��                                    BxX�Q�  �          A*=q�-p��У�@�ffBAz�Cwp��-p���{@ڏ\B�HCz�{                                    BxX�``  �          A*�\�-p���{@��HB=��Cw�R�-p����\@�ffB�RCz�3                                    BxX�o  �          A*{�<����{@��RB:
=Cv33�<����=q@��Bp�CyT{                                    BxX�}�  �          A*{�6ff�޸R@���B4�Cw���6ff� ��@ʏ\B
=Cz�=                                    BxX��R  �          A)�)����(�@�(�B0�Cy�H�)����@���B�HC|#�                                    BxX���  �          A)���'���(�@�z�B)\)Czz��'���H@��
Bp�C|�                                     BxX���  �          A(���7
=��R@�z�B*p�CxT{�7
=�(�@�z�B��Cz�H                                    BxX��D  �          A)��� ����p�@�(�B!�C{��� ���
�H@��A��HC}�H                                    BxX���  �          A)��7
=��Q�@�p�B�CyǮ�7
=��
@�=qA��\C{��                                    BxX�Ր  �          A(Q��$z����R@�z�B��C|
�$z��=q@�Q�A㙚C}޸                                    BxX��6  �          A)��*=q� ��@�(�B��C{�R�*=q��@�\)A�{C}��                                    BxX���  �          A)��-p���\@�
=B��C{�H�-p���@���A�33C}^�                                    BxX��  �          A*=q�7��  @\B\)Cz�{�7��{@�z�AθRC|��                                    BxX�(  �          A)��{�
=@ə�Bp�C}8R�{�@�33AٮC~�
                                    BxX��  �          A)p��   �\)@ƸRBp�C}��   �{@�Q�A�\)C~�                                    BxX�-t  �          A)���%���H@�
=B�HC|��%����@���A�(�C~33                                    BxX�<  
�          A)p�������@�33B  C}�����33@�(�Aϙ�Cp�                                    BxX�J�  (          A*�\��ff���@��HB  C�� ��ff��@�=qAˮC�Z�                                    BxX�Yf  �          A+������@���A��
C�P�����!G�@hQ�A�(�C�p�                                    BxX�h  �          A*ff����Q�@��B�C��{����{@�G�A�z�C�33                                    BxX�v�  �          A*�R��ff�
=@��HA�
=C�R��ff�(�@~�RA��C��                                     BxX��X  �          A+\)��G���@�z�A�z�C�<)��G����@���A���C���                                    BxX���  �          A+
=��=q��@�33B{C�����=q�33@�  A��C��                                    BxX���  �          A+
=���H�
�H@���B

=C�=q���H�p�@��RAŮC��\                                    BxX��J  T          A,Q쿺�H�ff@��B33C�쿺�H��\@�  A�C��
                                    BxX���  �          A,Q쿓33�=q@��B�C�R��33��R@�33A�{C��                                    BxX�Ζ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��<              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���   8          A*=q���H��H@�(�Bz�C������H��R@���AָRC�U�                                   BxX���  
�          A*{��33�
�H@��\B��C�����33��@�ffA�33C�q                                   BxX�	.  �          A*�R��ff���@��B=qC����ff��@��\A�  C�y�                                   BxX��  �          A*ff�^�R��\@�(�A�RC�Ff�^�R��@j�HA�33C�~�                                   BxX�&z  �          A*�\�����=q@���A�\)C��{������H@c33A�  C�
=                                    BxX�5   �          A,  ��p��z�@�ffB �C�"���p���\@\)A�ffC�y�                                    BxX�C�  �          A-G���{�\)@��B33C�xR��{�=q@���A�{C���                                    BxX�Rl  �          A-������ff@���Bz�C�,Ϳ������@�=qA�z�C��=                                    BxX�a  �          A-G������\@�=qBG�C��)����@�33A���C��\                                    BxX�o�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�~^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��   8          A+�
�����Q�@��A��C��������!p�@aG�A�33C���                                   BxX���  �          A+\)�}p��G�@��HA�C�⏿}p��33@u�A��C�'�                                   BxX��P  �          A+33�����
@���A�z�C��H���� ��@^�RA�  C��f                                   BxX���  �          A+33��=q��H@�=qA��HC�⏿�=q� (�@a�A�Q�C�8R                                   BxX�ǜ  T          A+�
����R@�{A�C�Y���� Q�@h��A���C���                                   BxX��B  �          A,z�0����
@��A�
=C�޸�0���!��@j�HA�G�C��                                   BxX���  �          A,Q�z�H�p�@�{A�G�C��Ϳz�H�   @x��A�p�C�33                                   BxX��  �          A,Q�L����@�
=A�ffC����L���!��@h��A�{C��q                                   BxX�4  �          A+�
�333�
=@�  A���C��{�333� ��@j�HA��C��                                    BxX��  �          A+��   �Q�@��A�{C�t{�   �!@`��A���C���                                    BxX��  �          A+���
=�(�@�p�A�\C����
=�!�@c�
A��C��                                    BxX�.&  �          A,z�\�@��\A��C��{�\�#\)@\��A�\)C��                                    BxX�<�  �          A,z�>��\)@���A�Q�C�g�>��$Q�@O\)A�(�C�`                                     BxX�Kr  �          A-G������(�@��HA��RC��q�����"�\@mp�A�Q�C�{                                    BxX�Z  �          A-���L����\@��
A�C�b��L���$Q�@]p�A�
=C�o\                                    BxX�h�  �          A.�R��ff�
=@�Q�B��C��;�ff��@��A���C��{                                    BxX�wd  
�          A/33��R�Q�@���B�C�
=��R� ��@�G�A��RC�=q                                    BxX��
  T          A/�
���R�@ə�B33C�=q���R��R@��RA��C��q                                    BxX���  �          A/�����(�@�ffB�
C�b������
@��\A�G�C��                                    BxX��V  �          A/���33�ff@��B�C��)��33��\@�ffA��HC���                                    BxX���  �          A0  �c�
�33@��B{C�
�c�
�p�@�=qA�{C�l�                                    BxX���  �          A0Q쿂�\���@�G�BC��{���\��R@�A�33C��                                    BxX��H  �          A0(�������@ҏ\B�C�uþ���
=@��RA���C���                                    BxX���  �          A0�׿+��
�R@ٙ�B33C��=�+��p�@�ffA�33C��                                    BxX��  �          A1p��
=�	�@�B�
C�Ϳ
=�G�@�=qA�{C�H�                                    BxX��:  �          A1���Ǯ��
@�\)B#
=C����Ǯ�Q�@�{A�p�C�\)                                    BxX�	�  �          A1��W
=�33@�A�C�g��W
=�!��@^�RA��C���                                    BxX��  �          A0Q쿊=q�
=@��
B��C����=q�"�H@z=qA��C��R                                    BxX�',  �          A1p���(��	�@ٙ�Bz�C�T{��(��Q�@�Aә�C��                                    BxX�5�  T          A1��p���@��B
z�C�����p�� ��@��
A�33C��                                    BxX�Dx  �          A1���������@�33A���C�
=�����*�R@?\)Az�RC�q                                    BxX�S  �          A.�R��  ��@�33B  C�N��  �!�@x��A�G�C���                                    BxX�a�  �          A,  ��33�
�R@��HB
�\C�� ��33��@�ffA�z�C�33                                    BxX�pj  �          A��������@���B�
C�R�����
�H@��HA�
=C��                                    BxX�  �          AQ��*�H���@�(�BG
=Cs� �*�H��Q�@��
B�Cxs3                                    BxX���  �          A�����p�@��B<�HCv�{����p�@�33B�\Cz��                                    BxX��\  �          A
�R��{�ȣ�@�B%{C�{��{���@�Q�A�p�C�Ǯ                                    BxX��  �          A	���0������@P  A���C�z�0���z�?���A*�RC���                                    BxX���  �          AQ�@�� z�Y����Q�C�k�@����\)���C�Ǯ                                    BxX��N  �          A  @8Q������\�<Q�C��f@8Q���{�X����=qC���                                    BxX���  �          A��@������R�\��(�C���@�����{��G����HC�B�                                    BxX��  �          A��@�  �
=�!G��t(�C��
@�  ��\)��p����C��                                    BxX��@  �          Ap�@�z����G
=��\)C�|)@�z���33��ff��33C��q                                    BxX��  T          A��@]p��\)��ff�2ffC�q@]p���z��b�\���HC��{                                    BxX��  �          A  @Y���{��Q��A��C��@Y�������j�H����C��                                    BxX� 2  �          A(�@n{�=q��  ����C��{@n{���@  ��G�C��\                                    BxX�.�  �          Ap�@g
=� ����U��C�\@g
=���p�����
C��3                                    BxX�=~  �          A��@xQ�����#33��(�C�7
@xQ����H�����مC�Z�                                    BxX�L$  �          A{@e�� ����\�iG�C��)@e���(��}p���=qC���                                    BxX�Z�  �          A�@9����׾�p��33C�!H@9���  �  �`��C�h�                                    BxX�ip  �          Ap�@q���Ϳ�
=�?
=C�=q@q���{�j�H��G�C��                                    BxX�x  �          A
=@aG��
�H�8Q���z�C��@aG����$z��~ffC�w
                                    BxX���  �          A
=@L(��녾B�\��
=C��@L(��	�
=�N�\C�(�                                    BxX��b  �          A�@E��
=�L�;��RC�}q@E������R�B{C��
                                    BxX��  �          A�@O\)�{>�p�@p�C��f@O\)��
��\)�C��=                                    BxX���  �          A  @a���ÿ���N�RC��f@a���� ���n�HC�f                                    BxX��T  �          Az�@Tz���R�
=q�K�C���@Tz��G��!��pz�C�K�                                    BxX���  �          A33@�ff��{�~�R��
=C���@�ff��(���(��	�C�u�                                    BxX�ޠ  �          A�@x����]p����\C�z�@x����z����R����C��                                     BxX��F  T          Aff@�����\)�W
=��G�C��@����љ���p����C�˅                                    BxX���  �          A33?�Q���
?�(�@��C���?�Q���׿J=q��{C��f                                    BxX�
�  �          Ap�?����
=?Y��@�(�C�e?�����\��33��Q�C�j=                                    BxX�8  �          A\)?��\�p�?!G�@���C�{?��\�(������
�RC�q                                    BxX�'�  �          A\)?G��(�?��HA0(�C��=?G���R��\)��ffC�~�                                    BxX�6�  �          AQ�?
=q�  @ffAW�
C��f?
=q�  =���?!G�C���                                    BxX�E*  �          A(�?h���
=@	��A]�C�  ?h���33>��?uC��=                                    BxX�S�  T          AG�?�(���H>���@#33C���?�(���׿�{�#�C��q                                    BxX�bv  
�          A�?5�33?}p�@�G�C�AH?5�33��ff���C�B�                                    BxX�q  �          A�\?(���  @$z�A���C�(�?(���?�@UC�3                                    BxX��  �          A?��G�@Q�AXz�C���?��G�=��
>��HC��                                     BxX��h  �          A��?���Q�@�AY�C���?���Q�=��
?   C���                                    BxX��  �          A����H��z�@��B C�k����H�z�@<(�A�=qC�H                                    BxX���  �          A{�   ��\)@�  B4
=Cvs3�   ��  @���B
=Cz^�                                    BxX��Z  �          A��/\)���@��
B@�Cs���/\)��p�@�{B��Cxff                                    BxX��   �          A�H�J=q���\@�\B^��ChB��J=q���R@�{B2��Cq
=                                    BxX�צ  �          A�
�5����@��BF=qCq���5�׮@�(�BG�Cw(�                                    BxX��L              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�>   7          A�\�G
=��33@�\BW33Ck�f�G
=��\)@��B(��Css3                                   BxX� �  "          A33�l�����H@�G�BS��Ce���l����
=@ʏ\B(
=Cn0�                                   BxX�/�  �          A\)�p������@�BY�Cc\�p����
=@���B.�\Cl��                                   BxX�>0  �          A���qG��e�@��RB_��C_z��qG�����@��B7  Cjn                                   BxX�L�  "          A33�@  ��@��
BW��Cmc��@  �ʏ\@�=qB((�Ct��                                    BxX�[|  T          A���*�H��\)@�Q�BO�HCr�q�*�H��33@\B\)Cx��                                    BxX�j"  �          A��1���p�@�z�BB  Cs�H�1���{@��B�Cx�)                                    BxX�x�  T          A�
�.{��\)@˅B(��Cw=q�.{����@��A�CzǮ                                    BxX��n  �          A���Fff��  @�  B#Ctz��Fff��G�@�G�A�\)CxJ=                                    BxX��  �          Ap��<����
=@�=qB"  Cus3�<����
=@��
A�
=Cy\                                    BxX���  "          A�H�333����@�{B!p�Cvff�333���
@�  A�
=Cy�)                                    BxX��`  
�          A(��.{��ff@���B�
Cw���.{��z�@�G�AθRCz�R                                    BxX��  �          A�\�@  ��(�@�
=B-(�Cr���@  ��ff@�z�A�=qCwY�                                    BxX�Ь  
�          A������@  @�  Bb
=CW�������Q�@��B<�Cd�H                                    BxX��R  T          A�R��=q�3�
@�(�Bc�CU���=q���
@�\)B?=qCb�H                                    BxX���  
�          A�������8Q�@��RB`�RCU���������@�G�B<=qCcQ�                                    BxX���  "          A33��Q��
=q@�ffBc��CM����Q��|��@޸RBD��C]0�                                    BxX�D  "          A����
=�c�
@�ffBk�C>�f��
=�1G�@�  BW��CRs3                                    BxX��  "          Ap�������
@�Bh��CC�����I��@�(�BQ�\CU�)                                    BxX�(�  T          Aff��z��*=q@�=qB_Q�CS5���z����@�ffB<ffCa�                                    BxX�76  "          A�����$z�@�RB`Q�CR�\�������@ӅB=�C`�                                    BxX�E�  
�          A�R���R�1�@�B^G�CUu����R��\)@θRB9�RCb�\                                    BxX�T�  '          Az������.{@陚BX{CR��������p�@�p�B5z�C`(�                                    BxX�c(  �          A������0��@�G�B_p�CT�3��������@�z�B;(�Cbc�                                    BxX�q�  "          A����1�@��B^{CU������\)@��B9(�Cc�                                    BxX��t  T          A
=��
=�G�@��HBZ�CN�f��
=�{�@�=qB:��C]W
                                    BxX��  �          A�ÿ�=q��p�@]p�A�
=C����=q�ҏ\?�33A~ffC�S3                                    BxX���  �          A	G�?z��p�?�
=A\)C��)?z���H�(����
=C��
                                    BxX��f  "          A\)�B�\���?}p�@��C�T{�B�\��׿�=q��C�T{                                    BxX��  "          A�Ϳ��
��(�@33A�z�C�aH���
�33>�\)?�(�C��\                                    BxX�ɲ  
�          A���aG����R@�{BCoff�aG����
@5�A��Crٚ                                    BxX��X  �          A{�r�\��ff@�33B�Ci޸�r�\�˅@y��A��Co.                                    BxX���  �          A z��������R@�=qB�HCgs3�������
@9��A���Ck�
                                    BxX���  
�          A�aG���
=@���B,CiJ=�aG���ff@�A�p�Co^�                                    BxX�J  '          @�{�I�����@�=qB2�
Ck��I������@��BQ�Cq�H                                    BxX��  �          @��R��33�QG�@�z�BtCo�{��33��z�@�33B@z�Cx��                                    BxX�!�  �          @��p���ff@�  BA�HCo�f�p���{@���BQ�Cu�                                    BxX�0<  �          @���{�&ff@�\)BgCb�=�{�}p�@�z�B9�Cn
=                                    BxX�>�  �          @陚�g
=�aG�@��
Bj�CA���g
=���@�ffBS=qCUs3                                    BxX�M�  
�          @���@e�`  ����
=C���@e�W
=��  �8(�C�t{                                    BxX�\.  
Z          @���@��|��?޸RA��C�}q@���Q�?0��A��C���                                    BxX�j�  
�          @�z��tz����@�B5
=CJ�3�tz��&ff@h��B�CV@                                     BxX�yz  �          @��\(�=�G�@�p�Bk�C2
�\(�����@���Ba(�CI\                                    BxX��   
�          @���33���H>�\)@��C�uþ�33��{��33�D  C�n                                    BxX���            @�G�����ff?.{@���C�z����������\C�z�                                    BxX��l  
-          @��H���"�\@l(�B9�CbL����X��@:�HB��Cjn                                    BxX��  �          @�(�� ����\@s33B@C^aH� ���K�@EBQ�Cg�q                                    BxX�¸  T          @����Dz�@���BLG�Ch\)����{@w
=B�
Cp��                                    BxX��^  T          @��H�\)�333@��RBG�CdaH�\)�vff@eBQ�Cm
                                    BxX��  
�          @�����G�@�{BP�\C`L����QG�@^{B#=qCj��                                    BxX��  �          @��H���h��@�33BC�HCup�����ff@aG�B�RCz�                                     BxX��P  
(          A{@���أ׿��z�HC��H@����������{C�Ff                                    BxX��  O          A��@�G����Ϳ����$��C��@�G������N�R��ffC�P�                                    BxX��  
�          AG�@��\����33�,z�C�R@��\��G��P  ���HC���                                    BxX�)B  
Z          A=q@ָR��녽��
�
=qC���@ָR��33��=q�$(�C�3                                    BxX�7�  �          A=q@ƸR�ə�?(�@{�C�L�@ƸR�Ǯ�����  C�n                                    BxX�F�  T          A��@�(��θR>8Q�?��C�U�@�(����ÿ�  ��
C���                                    BxX�U4  �          A  @�����z��&ff���RC��@�����
=���
��RC���                                    BxX�c�  �          A�
@�=q��=q�  �g�
C���@�=q�Ǯ��Q����
C���                                    BxX�r�  �          A  @��H��=q����x  C�f@��H���R���H�ظRC���                                    BxX��&  �          A�@���p��!���
=C���@���������Q�C��R                                    BxX���  T          A=q@����{�=q�{�C�޸@����=q��(���  C��R                                    BxX��r  �          A�R@�Q����
��Q��H��C���@�Q��˅�n�R���
C��                                    BxX��  �          A�@�p���(�?h��@���C�� @�p�����ff���C���                                    BxX���  �          Az�@����ff��\)��C�t{@����33�I����
=C��)                                    BxX��d  �          A33@�����=q��{��C���@��������;����\C���                                    BxX��
  
Z          A��@�����(����
�˅C���@����Ӆ�7����C��H                                    BxX��  	�          Az�@�33���8Q���ffC��@�33����(Q����C��f                                    BxX��V  
�          A  @�z��У׿��
�  C�5�@�z���(��L(���p�C���                                    BxX��  "          Az�@׮��Q쿜(���\C��\@׮���R�5���Q�C���                                    BxX��  �          A��@��H��\)�\(�����C��H@��H�����"�\�~=qC��                                    BxX�"H  �          A��@Å�޸R�s33��33C���@Å�θR�0����  C��{                                    BxX�0�  �          A@�=q��=q�^�R���C�P�@�=q���H�.{���C�T{                                    BxX�?�  
�          A33@��
��  ���\��
=C��@��
����AG����C�c�                                    BxX�N:  T          A�
@�{��׿J=q��p�C�"�@�{�љ��(Q���z�C��                                    BxX�\�  "          A
=@�(����
�p����=qC���@�(��˅�/\)��
=C���                                    BxX�k�  "          A�@������Ϳ�\)��\C���@�����33�:�H��(�C�'�                                    BxX�z,  T          A�@������ÿ�ff�=qC��H@������H����\)C�f                                    BxX���  
�          A\)@������ÿ�\)�z�C���@�������L������C�
=                                    BxX��x              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�)�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�8�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�G@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�U�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�d�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�s2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�ټ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��b              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�T              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�"�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�1�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�@F              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�N�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�l8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�z�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��v              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��h              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�*�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�9L              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�G�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�V�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�e>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�s�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��n              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�`              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�#�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�2R              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�@�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�O�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�^D              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�l�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�{�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��t              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��f              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�+X              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�9�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�H�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�WJ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�e�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�t�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��<              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX���  �          A��?E���Q�@c33A�33C��=?E���?��HAC��=                                    BxX���  c          AG�?\)��
=@��HA�C�9�?\)���?���AJ{C��)                                    BxX��.  
�          A�>�=q��p�@y��A�Q�C�
>�=q���?�33A:�HC���                                    BxX���  �          A\)@q����@+�A��
C�
@q�����?�\@^{C�T{                                    BxX��z  �          Az�@I����@J=qA��RC��f@I�����H?k�@ǮC��{                                    BxX��   
�          A(�@@  ��@]p�A��C�Y�@@  ���
?�p�Az�C�o\                                    BxX���  T          A��@:�H��  @_\)A��\C��R@:�H��ff?�p�AQ�C��                                    BxX��l  �          AQ�@8����=q@QG�A�ffC�Ǯ@8�����R?�  @�
=C��q                                    BxX�  T          A��@.{��(�@X��A�z�C�q@.{� ��?��@��HC�U�                                    BxX��  
�          A	@{��33@n{A�33C�K�@{�?�AQ�C�xR                                    BxX�$^  
�          A	G�@����
=@|(�AڸRC�1�@��� ��?�z�A2�HC�L�                                    BxX�3              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�A�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�PP              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�^�   5          A
=@9����=q@HQ�A��HC�z�@9����\?J=q@�ffC��=                                   BxX�m�  T          A��@<����  @(Q�A���C��
@<����ff>���@\)C�+�                                   BxX�|B  "          A	p�@:�H����@p�At(�C��)@:�H��33��Q�(�C�8R                                   BxX���  "          A�@�����
=��G��C��f@�����  �U���C��
                                   BxX���  
�          A
{@�z���z�n{���C�\)@�z�����9�����RC��\                                   BxX��4  
�          A�@W
=��\>��@N{C��@W
=��z��\�D��C�5�                                    BxX���  T          Ap�@P  ��(�>�z�@�
C��q@P  ���Ϳ���W�C�7
                                    BxX�ŀ  
�          A33@�����H�Q����RC�\@����Q��Dz����C��=                                    BxX��&  
�          A{@����녾�����
C��@������H��33C�                                    BxX���  "          A��@33���
>���@�\C�&f@33���
���R�`  C�ff                                    BxX��r  "          Az�?�=q��{?�\)A��C��H?�=q�����G���Q�C�u�                                    BxX�   �          A�
?�(���z�?�33A�C�%?�(����R�xQ��ٙ�C��                                    BxX��  �          A�R?ٙ���\)?�(�A_\)C�7
?ٙ����R�\�*=qC��                                    BxX�d  
�          A z�@3�
����?�\ALz�C��@3�
��\)��G��HQ�C�L�                                    BxX�,
  
Z          @��H@K����
�����33C�
=@K��׮����
=C��f                                    BxX�:�  T          @�p�@R�\���H���H�k�C��q@R�\��z��{��p�C��H                                    BxX�IV  �          @�=q@fff�ҏ\>�p�@7
=C�W
@fff���Ϳ����@��C���                                    BxX�W�  �          @�{@?\)��G�?z�@��HC���@?\)���Ϳ�  �4��C���                                    BxX�f�  
�          @�Q�@Y�����
?��\@��C���@Y����(��u���HC��
                                    BxX�uH  �          @�33@a��Ӆ?�
=A�HC��@a�����O\)���
C��R                                    BxX���  �          @�\)@C�
��(�?��A8z�C��)@C�
���ÿ����=qC�                                    BxX���  
�          @�
=@L����  ?�p�APz�C��\@L���޸R���
�=qC�XR                                    BxX��:  T          @�  @<(���\)?�Q�A1��C���@<(��ۅ�
=��G�C���                                    BxX���  T          @�@ff�߮@:=qA�{C�Z�@ff����?=p�@�z�C�Ǯ                                    BxX���  �          @��\@(Q���  @�A�C�J=@(Q���>�?xQ�C��3                                    BxX��,            A{@  ��z�@q�A�=qC��@  ��{?˅A4  C�%                                    BxX���  
�          A@�H��z�@a�A�z�C��@�H��(�?�{AffC��\                                    BxX��x  �          A Q�@33���@a�A�\)C�T{@33��?���A33C�y�                                    BxX��  
�          A�?����  @�{A�p�C�q�?����@�Ahz�C��\                                    BxX��  
�          A?�Q���Q�@���B G�C���?�Q���
=@Q�AtQ�C�aH                                    BxX�j  T          A ��?����θR@�G�B	�C��?�����Q�@��A�\)C�H                                    BxX�%  
�          A z�?�Q���p�@�{A��RC�)?�Q���33@33AlQ�C�C�                                    BxX�3�  
�          A{?�\)����@x��A��C��=?�\)���?�z�A<��C��                                    BxX�B\  T          AG�?�����@�33B��C���?������@p�A~�RC���                                    BxX�Q  �          @���?�����p�@��RB�C�?������
@��A~�HC�J=                                    BxX�_�  
�          @�=q@33����@��B	�\C�s3@33�ᙚ@p�A���C��                                    BxX�nN  �          @�\)?������
@��\BQ�C�O\?������@4z�A�=qC�33                                    BxX�|�  "          @�
=?���љ�@�(�B��C�� ?����@G�A�z�C��=                                    BxX���  T          @���?}p����@�\)B\)C�g�?}p����
@
�HA�  C��
                                    BxX��@  �          @�?h���ȣ�@~{B =qC�)?h�����?��HAs33C��)                                    BxX���  �          @�(�?Q����@|(�BffC��=?Q���G�?��HAw\)C�Q�                                    BxX���  "          @�G�?�p���ff@s�
A��RC�Ф?�p���G�?�=qAb{C��                                    BxX��2  T          @�\)@�Q����@L��A�=qC�ff@�Q����?��RA9G�C���                                    BxX���  
�          @�\)@1���=q@R�\A�{C�K�@1��أ�?�\)A*=qC�(�                                    BxX��~  �          @���@����=q@Tz�A�G�C���@������?�33A0(�C��                                    BxX��$  
�          @�G�@%��˅@Dz�A���C�
=@%���\)?�=qA�C�%                                    BxX� �  �          @���@��˅@_\)A�\)C�:�@���?�(�A2{C�=q                                    BxX�p  �          @�
=@Q��Å@_\)A�ffC��H@Q��ۅ?ǮAAG�C���                                    BxX�  �          @�ff@P����@EA�ffC�e@P����=q?�(�A\)C�/\                                    BxX�,�  "          @�\@�G���33@0  A�ffC��
@�G����?}p�@�G�C�j=                                    BxX�;b  �          @��
@�G���z�@3�
A��C���@�G��ƸR?��@��C�T{                                    BxX�J  �          @��
@�����=q@3�
A�33C�XR@�����z�?}p�@�RC��                                    BxX�X�  �          @�=q@�����R@
=A�C���@������?
=@�Q�C���                                    BxX�gT  "          @�G�@��\���
?��RAuC��3@��\���R>��@G�C���                                    BxX�u�  �          @���@�����{?�(�As33C���@�������>���@G�C���                                    BxX���  �          @��@^�R���@$z�A��C��\@^�R�љ�?0��@�=qC�                                      BxX��F  �          @�ff>�33��(�@qG�A�(�C��>�33��ff?޸RAYp�C�`                                     BxX���  V          @��?�  ���H@HQ�A�33C�p�?�  ��
=?���A�C���                                    BxX���  
�          @�  @�
��ff@B�\A�\)C��{@�
���?��
@�(�C��                                    BxX��8  "          @�������6ff@(��A���CQ�������_\)?ٙ�AxQ�CV�
                                    BxX���  �          @�{�ȣ׿��@�A�{CC�f�ȣ���?�z�A^ffCH�R                                    BxX�܄  �          @�\�Å�Q�@8Q�A�=qCG33�Å�7�@��A��RCM33                                    BxX��*  T          @��
���
���@N{A���CIW
���
�B�\@(�A�p�CPc�                                    BxX���  T          @�p���녿�p�@L��A�z�CF�{����5�@\)A��\CM�3                                    BxX�v  T          @�  ��녿���@]p�A�
=CEff����0  @1�A�\)CMQ�                                    BxX�  �          @����
=�s�
@Z�HA�33C\s3��
=��z�@
=qA�{Cb\                                    BxX�%�  �          @�\�e�����@^{A���Cj�=�e���=q?�{At  Cnc�                                    BxX�4h  
�          @�����{��33@���Bp�Cb���{���H@'
=A��\Cg޸                                    BxX�C  "          @�  �������@}p�BG�Cd������\)@{A���Ciz�                                    BxX�Q�  
Z          @���ff��@vffB  Cb� ��ff��33@��A��Cg�H                                    BxX�`Z  T          @�=q�g����R@|��B
=Ci�f�g���z�@A�Cnu�                                    