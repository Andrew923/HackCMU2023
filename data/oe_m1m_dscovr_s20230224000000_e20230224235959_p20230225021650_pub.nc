CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230224000000_e20230224235959_p20230225021650_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-25T02:16:50.176Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-24T00:00:00.000Z   time_coverage_end         2023-02-24T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxh�   �          A�R?��H<#�
@�ffB�G�>���?��H����@�33BbC��                                    Bxh��  �          AQ콣�
?��HA�B��
B��ͽ��
�Dz�@��B��RC�:�                                    Bxh�L  �          A�?J=q?��
A�HB�8RBz�?J=q�7
=@�=qB�u�C���                                    Bxh��  T          A(�?�p�?�33@�z�B�p�B�?�p��:=q@��B���C�|)                                    Bxh��  �          A	�?z�H�n{A��B��{C�@ ?z�H����@�p�BQ�
C�}q                                    Bxh�>  �          A��=�G�?}p�A�RB��B���=�G��\(�@�  B�{C��3                                    Bxh��  
�          A�\@�G���z�@	��Ar�RC���@�G���(����
��RC�'�                                    Bxh�  �          AG�@�(�����?(��@��C�7
@�(������   ��
=C�w
                                    Bxh�0  �          Ap�@e���p�@�\)BqQ�C��@e���G�@�(�B��C��                                    Bxh�  �          @��@Fff��ff@���B�Q�C���@Fff��\)@��B)��C��\                                    Bxh|  �          A=q@P�׿���@�Bx��C���@P�����\@�{B�HC��{                                    Bxh$"  �          A��@�p��(��@ƸRBN
=C��3@�p���@�  A�=qC��H                                    Bxh2�  �          A�R@�����@���B��C���@�����@�A{�C��                                    BxhAn  �          A�R@�z���@�
=B  C�� @�z����H@�A_\)C���                                    BxhP  
Z          Aff@�33���\@��B��C�S3@�33��Q�@G�A`  C�W
                                    Bxh^�  "          A=q@����=q@��RB)G�C���@����  @ ��A��C�O\                                    Bxhm`  �          @�p�@�Q���{@��\B!\)C��@�Q����@�Az�\C��                                    Bxh|  
�          A@w��B�\@�\)BS33C��=@w����@�G�A�{C���                                    Bxh��  
(          @��@j�H�>{@�ffBWG�C�� @j�H���H@���A�C�3                                    Bxh�R  
�          Ap�@XQ��r�\@���BL(�C���@XQ���ff@_\)Ạ�C�Ф                                    Bxh��  �          A�@�z��:=q@�z�BNp�C�p�@�z���  @���A��HC��                                     Bxh��  �          A\)@��\�/\)@�G�BT=qC��@��\��@�  A�  C��R                                    Bxh�D  �          @�p�@��H�P  @�{BBC��@��H���@^{A�ffC���                                    Bxh��  T          A{@���?u@ӅBU�A5@����!G�@�{BC�C�%                                    Bxh�  
�          A	@���@%@У�BHQ�A�R@�������@�{BZ��C���                                    Bxh�6  �          A
=@��H@�33@���B�B9�H@��H@@�Q�BK�A��R                                    Bxh��  "          A�R@���?��@�ffBO��AO\)@����
=@��BB�C�~�                                    Bxh�  "          A@�  �\(�@��
Ba=qC��3@�  ��
=@��B&��C�h�                                    Bxh(  "          A�@�Q��G�@�Q�BUQ�C�� @�Q��s33@��B&�HC�l�                                    Bxh+�  "          A	G�@\)@#33@�p�BC\)B�@\)�
=q@��RB_��C�                                      Bxh:t  T          AG�@���@�z�@��B��B]�H@���@��@���Bl�RA�                                    BxhI  �          A��@�  @��H@љ�B9  B#(�@�  >aG�@�
=Bi�\@-p�                                    BxhW�  �          A��@��@J�H@θRB8�A���@����@�p�BTp�C�4{                                    Bxhff  �          A�@��?h��@�z�B^G�A%@���<��@��HBG  C��{                                    Bxhu  "          A	��@���>��@�Q�BOz�?�z�@����Q�@��B-�C�"�                                    Bxh��  �          A�H@�=q?�@�p�B:�Av{@�=q��
=@�33B8(�C�33                                    Bxh�X  T          A\)@B�\?�p�@�\)B�{A��@B�\�0  @�z�Bw
=C���                                    Bxh��  T          A
�R@"�\?h��A  B���A�{@"�\�W�@�\Bs��C�|)                                    Bxh��  �          A\)@	��?z�A�\B�aHAr�R@	���o\)@��BqG�C��\                                    Bxh�J  �          AG�@0��    AB�z�=u@0�����R@�
=B\�\C��                                     Bxh��  "          A�R@R�\�G�A(�B��{C�P�@R�\���R@޸RBEC���                                    Bxhۖ  
�          A��@qG��!G�A�
B��\C�Ff@qG����@��BC�
C�f                                    Bxh�<  _          A�@�=q��G�@��BmffC�H�@�=q���@ȣ�B)�C���                                    Bxh��  T          A33@�Q��(�@���BI�C��@�Q�����@��B��C���                                    Bxh�  
(          A33@�p��/\)@���B@�C��@�p�����@�ffA��C��                                    Bxh.  �          A
=@�Q��`��@߮B?�C�(�@�Q���G�@�G�A�Q�C�ff                                    Bxh$�  �          A��@�Q��p  @�33B6z�C�AH@�Q����@u�A�C�]q                                    Bxh3z  �          A(�@����ff@���B+�C�H@����R@Dz�A��C�                                      BxhB   "          A{@hQ����
@�ffB*(�C��@hQ���(�@��A�\)C���                                    BxhP�  �          AQ�@u��Q�@޸RBJ  C��@u��@xQ�A�{C�{                                    Bxh_l  �          A  @@������@ָRBAp�C�&f@@����=q@N{A�z�C��f                                    Bxhn  
�          A z�@��?��@�\)B�(�A��@���@ָRB~�C��\                                    Bxh|�  "          A�
@	��@	��@�B��B4
=@	����{@�  B��\C��\                                    Bxh�^  
�          A�@4z�@�Q�@��B0G�Bm��@4z�?�Q�@��B�u�A�G�                                    Bxh�  
�          A��@G
=@��@���B��Bq��@G
=@�@�=qB{(�B	
=                                    Bxh��  �          AG�@J�H@��@�Q�BN�BP33@J�H>��@�p�B�  @�
=                                    Bxh�P  T          A��@\(�@l��@�  BP��B<z�@\(��#�
@�\)B���C��f                                    Bxh��  
Z          A�R?k�@�A z�B��B�#�?k���A ��B���C�Ǯ                                    BxhԜ  T          A��@
�H@�=q@أ�B^��Bw�R@
�H>��@���B�(�@���                                    Bxh�B  �          A(�@7�@�p�@�=qB?��Bi��@7�?��\@��HB�=qA���                                    Bxh��  �          A(�@C33@���@���B�Bt33@C33@
=q@��B{��B
=                                    Bxh �  T          A  ?\@��\@�{B;��B�.?\?�@���B��qB>�\                                    Bxh4  
�          A��?�@�z�@�B�
B��q?�@0  @�B�Ba                                      Bxh�  �          A�?�z�A�?�\)AB��f?�z�@��
@�\)B�
B�Ǯ                                    Bxh,�  �          A
=@Q�@陚>�=q@�B�z�@Q�@�ff@w
=A�(�B��                                    Bxh;&  "          A��?�G����
@�(�B�{C��f?�G����@\B>�HC�Ff                                    BxhI�  �          Ap�@E@��@�Q�Bv�RB\)@E�\@�B���C��H                                    BxhXr  T          Aff@Fff@U@�\)Bc\)B<ff@Fff�\)@��RB��=C��R                                    Bxhg  
�          A	p�?�\)�˅@�B�� C��f?�\)��=q@�33BAG�C���                                    Bxhu�  	�          A��>����r�\@��B��C���>�����  @��B�C�P�                                    Bxh�d  "          Ap�=��
�{�@�(�B~  C���=��
��=q@���B33C�S3                                    Bxh�
  �          A=L���n�R@�
=B�C�c�=L����{@�=qB��C�4{                                    Bxh��  T          Ap�?E��N�RA�B�Q�C��?E��ٙ�@�p�Bz�C�7
                                    Bxh�V  T          A�@G���G�@�{B��C�H@G����@�BM�C��R                                    Bxh��  �          A�
@��
?��@�=qB[ffAD(�@��
�%�@�p�BI�
C��                                    Bxh͢  	�          A\)@H�ÿ��A ��B�
=C�xR@H�����@�ffB?�C�N                                    Bxh�H  �          A�@g���  @���BffC�~�@g���{@�p�B4�HC�"�                                    Bxh��  �          A�
?�z��.{A��B�z�C���?�z��˅@�ffB$��C�B�                                    Bxh��  "          AQ�?���|��@�ffBw�C�\?����\)@�  B
=C�|)                                    Bxh:  "          A��?�(��5A�B�ǮC���?�(��У�@�  B#C�P�                                    Bxh�  T          A?У��Z=q@�\)B�Q�C��=?У�����@���B�C���                                    Bxh%�  T          A\)������p�@��B1�RC�  �������@\)Ap(�C��q                                    Bxh4,  �          AQ�>���(�@���B�C���>��
�H?aG�@���C��H                                    BxhB�  �          A�?�녿^�R@���B�  C��{?�����\@�33BP(�C�>�                                    BxhQx  	�          A33?\>\)AffB�Ǯ@�{?\�xQ�@��BoG�C��3                                    Bxh`  �          A{?���p�A ��B�ffC��\?���p�@�G�B:�C���                                    Bxhn�  �          A�H?=p����@��B���C��R?=p���33@�  B,  C���                                    Bxh}j  
�          A��?����p�@�z�BXQ�C��\?����@g�AѮC�p�                                    Bxh�  �          A�?����#�
@�(�B�#�C�� ?�����
=@��RB 33C��                                    Bxh��  "          A�@
=q�L��@�Q�B{{C��)@
=q��
=@�\)B��C�9�                                    Bxh�\  
�          A�@\)�˅@��B�C���@\)��\)@\B9�\C��H                                    Bxh�  �          A\)@vff>�
=@���By�@�ff@vff�P  @��BS��C��=                                    Bxhƨ  "          Aff?�\�B�\A�RB�ffC��=?�\����@�G�Be=qC��=                                    Bxh�N  "          A�@33�333A (�B��3C���@33����@��BU{C�9�                                    Bxh��  "          A�\@4z�c�
@���B�Q�C�<)@4z����@�33BG33C���                                    Bxh�  S          A  @�33�8Q�@�Bs�C���@�33�p  @��HBC33C���                                    Bxh@  "          A33@�=q?У�@�  Bb\)A�\)@�=q��p�@��B]�HC���                                    Bxh�  T          A
{@���@^�R@���B0�B
�
@���>.{@�33BV��?�z�                                    Bxh�  �          A�H@�@ff@��
B�G�BA�
@���33@���B�ffC��                                     Bxh-2  "          A
=@J�H@U@��Bc{B9��@J�H���H@�Q�B��C��3                                    Bxh;�  "          Aff?�{@��\@�Q�BD�\B�Ǯ?�{?�z�@�{B�=qBff                                    BxhJ~  T          A�@\(�@\)@�Q�BL{BE  @\(�>���@�(�B�B�@�G�                                    BxhY$  �          A��@��@W
=@�
=B-(�Bff@��>.{@�z�BR��?�33                                    Bxhg�  
�          A�\@A�@	��@陚B|(�Bp�@A녿�33@�B�.C��                                    Bxhvp  T          A��@�=q@@��RB=�\A��@�=q��ff@�\)BI�C�#�                                    Bxh�  "          A��@љ�@7�@��BG�A�G�@љ�>��@��RB=q@�\                                    Bxh��  �          Az�@���@*=q@��A��A�G�@���>B�\@���B�H?��                                    Bxh�b  �          A
=@�{?�=q@�p�B3  A���@�{��  @�=qB8C���                                    Bxh�  
�          A(�@���@�\@���B!�A�=q@����&ff@�{B1Q�C��                                    Bxh��  ,          A
=@��?�
=@�
=B'(�A�G�@�녿p��@�\)B0�HC�ff                                    Bxh�T  �          A?(����ff@׮B[ffC��)?(����@h��A�  C���                                    Bxh��  �          A�\@l(�@l��@�Q�B?33B4Q�@l(�>��@ڏ\Bu�R@陚                                    Bxh�  �          A�H@�z�@���@�
=B!\)B>�@�z�?�p�@�Q�Bc�A��                                    Bxh�F  �          A��@�
=@���@��
B\)B5�
@�
=@p�@�
=BG�
A�                                      Bxh�  �          A��@��R@��@�  B(33B9�\@��R?��R@�{Bf��A�
=                                    Bxh�  T          A�\@��
@�{@y��A�=qB%ff@��
@{@�G�B-ffA��                                    Bxh&8  �          A�@�z�@��?�  A/�B�@�z�@C�
@W�AˮA���                                    Bxh4�  �          A��@�  @�Q�>�?h��B�@�  @k�@	��Az{A���                                    BxhC�  "          AQ�@�
=@���?\(�@�
=Bz�@�
=@�Q�@J=qA�=qA�Q�                                    BxhR*  �          A�@���@�p��#�
��Q�BQ�@���@��@{A�ffB                                     Bxh`�  �          A�R@љ�@���@�\A�=qB=q@љ�@,(�@�=qA��HA���                                    Bxhov  
�          A�\@��H@�{@
=qAw
=B@��H@)��@x��A�A�p�                                    Bxh~  �          A�@ۅ@U�@\)A���A��@ۅ?޸R@q�A�RAc�
                                    Bxh��  T          A Q�@��@�@,��A��HA��
@��?Y��@a�A�
=@�(�                                    Bxh�h  "          A z�@�p�>�ff@���B;�\@�G�@�p��
=@��B(��C�ff                                    Bxh�  T          A ��@U��W
=@�  Bl�C���@U��^{@��B0G�C��f                                    Bxh��  �          A�?�(��^�R@�\)B|��C�p�?�(����
@�B=qC�B�                                    Bxh�Z  
�          AQ�?����(��@�{B��\C�f?������H@��B+�C���                                    Bxh�   �          A	G�@�G���@�\)B]�\C�(�@�G��n{@�p�B1�C��                                    Bxh�  T          A�R@��H>�(�@��BPG�@��\@��H�.�R@�\)B8  C��                                    Bxh�L  T          A��@�
=@8Q�@�ffB.�HA��H@�
=�#�
@�(�BJ�
C�                                      Bxh�  �          A�@��
@\)@�(�B3\)A�
=@��
�z�@˅BF�RC���                                    Bxh�  �          A
=@��@�H@�33B&�A�
=@�녿   @\B8�C��                                    Bxh>  �          A�@�
=@�@���B#(�A��R@�
=���@�
=B3��C�]q                                    Bxh-�  �          A
�H@ۅ@XQ�@�  A��HA�@ۅ?���@��
B�\AG�                                    Bxh<�  �          A33@ᙚ@Dz�@z=qA�A�=q@ᙚ?^�R@���B
Q�@��                                    BxhK0  �          A33@�33?�(�@���B��A:�H@�33�c�
@���B\)C�k�                                    BxhY�  �          A(�@�  ?�ff@��B��AG\)@�  �k�@���B�C�B�                                    Bxhh|  �          Ap�@���?޸R@l��A�
=AI@����L��@��HA�p�C�E                                    Bxhw"  �          A�@�z�?��R@l(�A�=qA�R@�z�!G�@uA���C��3                                    Bxh��  |          A�Ap�?�@9��A��AL��Ap�>�=q@Z�HA�33?��                                    Bxh�n  �          A�A�
@z�@W
=A��Atz�A�
>�@��A��@N�R                                    Bxh�  �          A  AQ�@
�H@0��A�  A^=qAQ�?!G�@]p�A��@��H                                    Bxh��  �          A(�A�
?��@N{A�\)A)A�
�u@g
=A��C�Ф                                    Bxh�`  T          A{A
=q?��\@W
=A�Q�A=qA
=q��G�@dz�A�33C���                                    Bxh�  
(          A=qA��?s33@C33A�Q�@�(�A�Ϳ
=@H��A��HC�R                                    Bxhݬ  �          A��A��?���@>{A��AIG�A��>��@_\)A�(�?�ff                                    Bxh�R  	�          A��A��@s33@#�
A\)A�
=A��@�@���A�\)Apz�                                    Bxh��  �          A  @�{@{�@@  A�\)A�z�@�{@@�\)A���Ar�R                                    Bxh	�  "          A�@��@��@��A��B��@��?Ǯ@��B)��AVff                                    BxhD  T          A�@�p�@�G�@Y��A�{B�R@�p�@*�H@��BQ�A�ff                                    Bxh&�  
�          A��@���@��?�A>=qBff@���@y��@��A�\)A��
                                    Bxh5�  
�          AQ�@���@��@S33A��B
=@���@C33@��Bz�A��\                                    BxhD6  "          A��@�=q@��R@k�A��HB(G�@�=q@W
=@���B�A�{                                    BxhR�  
�          A�@�(�@��H@p�A\��B
p�@�(�@`  @��A�ffA�
=                                    Bxha�  �          A��@�@�{@W
=A���A���@�@��@���B{A�(�                                    Bxhp(  �          A��@�  @��@
=AS
=B�@�  @]p�@�p�A�Q�A�                                      Bxh~�  
�          A�H@�@���?�\)A#
=B�\@�@n{@p  A�\)A�\)                                    Bxh�t  T          Ap�@��@��@/\)A���B�@��@AG�@���A�
=A���                                    Bxh�  �          A��@��
@�G�@
=AqB33@��
@Y��@��A�p�A�(�                                    Bxh��  
�          AG�@ٙ�@��R?�Q�A,��B (�@ٙ�@���@��
A�
=BG�                                    Bxh�f  �          Az�@��H@�G�?&ff@�p�B!{@��H@�33@N�RA��RBff                                    Bxh�  �          A  @�{@�z����z�HBz�@�{@��@�ApQ�B\)                                    Bxhֲ  �          A(�A ��@z�H?���AD(�AυA ��@&ff@aG�A���A���                                    Bxh�X  T          Aff@�(�@=q@�
Az{A�(�@�(�?��@I��A���@���                                    Bxh��  r          A	�@�\)�E�@�33BAz�C�|)@�\)��(�@�  A�(�C�                                    Bxh�  T          A��@<���Dz�@�{Bp��C��@<�����
@�\)BQ�C�ٚ                                    BxhJ  �          A
�R@G
=����@ٙ�BP�C��@G
=��z�@��A�(�C�%                                    Bxh�  "          A33@�\)�Z�H@�\)BC��C���@�\)��(�@�  A�p�C��H                                    Bxh.�  "          A33@��R�U�@�G�B=\)C��@��R��ff@w�A��C��                                    Bxh=<  �          A�\@��=�Q�@�\)B�?\(�@���
�H@�\)BffC�                                      BxhK�  �          A\)@Ǯ@W�@���B\)A��H@Ǯ?s33@��B%��A
�H                                    BxhZ�  T          A��@׮@}p�@A�G�A�@׮@�R@xQ�A��HA�\)                                    Bxhi.  T          A{@�ff@!�@��\A�p�A��@�ff>��@�G�B�@b�\                                    Bxhw�  T          A�@˅@]p�@��A�Q�A�Q�@˅?�G�@�{B��A2�R                                    Bxh�z  "          Aff@�G�@���@}p�A�\)BG�@�G�@��@���B"�HA�ff                                    Bxh�   
�          A
�H@�G�@�  @qG�AΣ�B{@�G�@ff@���BG�A�Q�                                    Bxh��  "          A�@�(�@33@��\A�=qAx��@�(�    @�=qA��<#�
                                    Bxh�l  �          A�
@���[�@���B��C��@������@'�A��RC��                                    Bxh�  �          AG�@���'�@�ffBz�C�Q�@�����@P  A��C��f                                    Bxhϸ  �          A@����@�(�B�
C��3@���Vff@�=qA�RC�:�                                    Bxh�^  T          A�@�33>���@�B	(�@S33@�33�޸R@��A��C��H                                    Bxh�  "          A{@�z�>\)@���B�?�(�@�z��Q�@��\BQ�C��                                    Bxh��  T          AQ�@ᙚ��@�p�A�Q�C���@ᙚ��@`  A�  C��                                     Bxh
P  T          A�H@θR��=q@��B33C�>�@θR�HQ�@w�A�ffC��                                    Bxh�  T          A\)@�녿�z�@��
B
33C��@���Tz�@`��A�{C��3                                    Bxh'�  �          A��@�����33@�33B!�
C�aH@����vff@��A�{C��H                                    Bxh6B  T          A��@�33�.{@��BG�C�@ @�33��
@���A�\)C�\)                                    BxhD�  �          A(�@�  ��@�B
z�C��@�  �"�\@}p�A��C���                                    BxhS�  �          A(�@��Ϳ��@��A�z�C���@����E�@R�\A�(�C��
                                    Bxhb4  T          A@�?
=q@i��AиR@���@�����@`��A�Q�C��\                                    Bxhp�  �          A
=@�p�?��H@~{A�z�AV=q@�p��aG�@�=qA�=qC�q                                    Bxh�  T          AG�@��?���@y��A�  AI@����\)@��RA�\)C�޸                                    Bxh�&  "          A�
@�z�?�@hQ�A��
Ag\)@�z�=L��@�=qA홚>�(�                                    Bxh��  T          A��@��@G�@�{A�(�A��@��>L��@�Q�B��?�33                                    Bxh�r  
�          A�@޸R@�@�A�A���@޸R>W
=@�Q�B	Q�?��H                                    Bxh�  �          A�@�\)?��
@�Q�B�A&=q@�\)�Tz�@��
BG�C���                                    BxhȾ  T          A�@ȣ׿�p�@��B�
C�K�@ȣ��j=q@dz�A��C��q                                    Bxh�d  T          A�\@�G��/\)@��B=qC�=q@�G���=q@HQ�A��C�H                                    Bxh�
  T          A
=@�Q��2�\@�p�B�C���@�Q�����@[�A�z�C�˅                                    Bxh��  �          A�
@��׿\@��B)=qC��=@����u@�A�=qC��H                                    BxhV  �          A�@����h��@���B6��C��@����\(�@���B�C���                                    Bxh�  �          A
{@�33=���@˅BB�?��@�33�+�@���B+�C�:�                                    Bxh �  T          A�
@�p���R@���B+C�%@�p��C�
@�B\)C���                                    Bxh/H  
Z          A
�H@�����@�
=BM=qC��3@������@�z�B�C�
                                    Bxh=�  �          A
=q@�  �ff@�Q�BD{C�q@�  ���R@��RBffC�                                    BxhL�  �          A�
@����H@�Q�B.z�C�q�@��l(�@�z�B�C��3                                    Bxh[:  �          AG�@��>��@�p�B\)?��\@�����@�ffB	Q�C�(�                                    Bxhi�  �          A��@���@ʏ\B6�C��@�����
@��BC��q                                    Bxhx�  �          A�H@�z�?���@�G�B"��A�\@�z´p�@��RB �C���                                    Bxh�,  "          A
=@�(���z�@�(�BE�\C���@�(��l(�@�G�B=qC�!H                                    Bxh��  
�          A�@���e�@��BL��C���@����ff@�ffB �C�*=                                    Bxh�x  T          Ap�@��L(�@���BH��C�� @���G�@�Q�B\)C�y�                                    Bxh�  T          A
=@������
@�G�B5�HC��@������@l(�AΣ�C�3                                    Bxh��  �          A��@U�����@�z�BD��C���@U��ָR@x��A�p�C�4{                                    Bxh�j  T          AQ�@p���'�@ٙ�B_��C���@p������@���B�C��q                                    Bxh�  �          A  @z�H�QG�@�BRffC��@z�H���@�z�B�C�H                                    Bxh��  �          Az�@��
�\(�@ҏ\BK�C��@��
��p�@��A��C�n                                    Bxh�\  �          A�R@=p�����@�Q�BM33C�B�@=p����@��A�33C��R                                    Bxh  �          A�\@g
=���@���B2�
C��f@g
=�ٙ�@U�A���C��q                                    Bxh�  �          A=q@
=��ff@�=qBDG�C��q@
=���@hQ�A�\)C�)                                    Bxh(N  
�          AQ�?�����
=@��HB%��C��?����   @�A�C��=                                    Bxh6�  �          A	�>\���
@�33B�\C���>\�(�@33A^ffC�T{                                    BxhE�  
�          A	p�>�G���@���B�C�� >�G��z�?��HAT��C���                                    BxhT@  �          A	�?333���@˅B@ffC��f?333���@XQ�A�Q�C��\                                    Bxhb�  T          A(�?:�H�}p�@�
=Bv�\C�@ ?:�H�׮@�33B�HC�)                                    Bxhq�  T          A�
?W
=���\@�BQ�C���?W
=��p�@{�A��C�:�                                    Bxh�2  "          A  ?�  ��p�@��BDQ�C�z�?�  ��@c33Ař�C��f                                    Bxh��  T          A�?�=q��Q�@�=qBA�\C���?�=q����@[�A�p�C��                                    Bxh�~  T          A�H?�����@���Bd{C��H?�����
@���B=qC��                                    Bxh�$  T          Ap�?޸R�r�\@���Bo
=C�\)?޸R��@��B��C���                                    Bxh��  �          A�R?����@���B-ffC�&f?���{@5�A��HC��                                    Bxh�p  �          A=q?�
=���@�  B$�C��?�
=���@   A�G�C���                                    Bxh�  T          A�?�����H@�  B:�\C��?����33@N�RA��\C�:�                                    Bxh�  T          A�?�z��\(�@�Bx��C��f?�z���(�@��BG�C��3                                    Bxh�b  S          A?����(Q�@�Q�B��
C��?�����Q�@��B5G�C�                                    Bxh  �          A z�@/\)����@�33B��C��
@/\)����@�(�BI�C�L�                                    Bxh�  T          A��?��,(�@�33B��{C��{?����@��\B/33C��)                                    Bxh!T  �          AG�?�=q�G
=@��B��RC�z�?�=q���\@��HB%C���                                    Bxh/�  �          @�z�@9����\)@�
=B=qC��H@9����33?�An�\C��R                                    Bxh>�  "          @�z�@Fff���@��HBA�C�aH@Fff��{@g
=A�=qC�P�                                    BxhMF  
�          @�z�@e��s�
@��B@
=C��R@e����@n�RA�  C��3                                    Bxh[�  "          @�ff@��H��
=@��BJp�C�'�@��H�c�
@���B �C���                                    Bxhj�  
�          @��@>{�Tz�@�BX��C��\@>{���@���B��C�3                                    Bxhy8  �          @��\@`  �%@�ffBRQ�C��q@`  ��ff@�z�B�HC�Z�                                    Bxh��  
�          @���?��Mp�@�33Bk\)C��\?���p�@�  B��C�8R                                    Bxh��  	�          @��R?�=q�e@ָRBsG�C�]q?�=q��Q�@��Bp�C��                                    Bxh�*  �          @��?���8��@޸RB��C�Q�?����\)@�B+p�C��                                     Bxh��  "          @���?��H�o\)@�  Bi  C���?��H���@��B33C��=                                    Bxh�v  �          @��
?��u�@���B^��C�O\?�����@�Bz�C��\                                    Bxh�  
�          @��H?�p��~�R@ƸRB\G�C�?�p�����@�=qB�HC��)                                    Bxh��  T          @�G�?�=q�r�\@�
=BbG�C�O\?�=q��\)@��B�C�e                                    Bxh�h  �          @�33�#�
���@�ffB\�C��3�#�
��ff@{�A���C��
                                    Bxh�  
�          @�@Tz��S33@�  BKz�C��{@Tz����\@{�B (�C��                                    Bxh�  
�          @�(�?E���z�@�33BR�HC��R?E����
@eA��
C�p�                                    BxhZ  �          @�p�����{@��
B^33C��3���ə�@z=qA��HC��                                    Bxh)   �          @��>��
����@�  BaffC�/\>��
���@��B��C�p�                                    Bxh7�  
�          @�{��33����@�  BY�C��=��33��=q@r�\A�\)C�k�                                    BxhFL  "          @��ÿ
=����@�33BV\)C���
=��Q�@i��A�\)C�J=                                    BxhT�  T          @�Q쿧����H@��BV�C|G�������H@o\)A���C��                                    Bxhc�  T          @�׿�����H@��BD(�C~G������z�@Mp�A�\)C�J=                                    Bxhr>  T          @��������@�G�B<G�C�� �����Q�@<��A�Q�C�C�                                    Bxh��  "          @�G��#�
��@��\B<{C�G��#�
��z�@<(�A�{C�:�                                    Bxh��  "          @�p�������(�@�=qBB�HC��������{@K�A�33C��
                                    Bxh�0  �          @�{�z����\@��
BDp�C���z����@P  A�G�C���                                    Bxh��  
�          @�=q�^�R��Q�@�33B?
=C��^�R���@J=qA��
C�XR                                    Bxh�|  "          @�z῀  ��=q@�G�B6�C�lͿ�  �׮@7
=A�  C��=                                    Bxh�"  �          @��?E����@���BNffC��\?E�����@]p�A��C�|)                                    Bxh��  T          @��?����^�R@�\)Bl�\C�j=?�������@��B�RC��R                                    Bxh�n  
�          @�G�?Ǯ�u@�\)B]{C��?Ǯ��(�@\)B{C�g�                                    Bxh�  �          @�z�@\)����@�  BM=qC���@\)��ff@mp�A�=qC�Q�                                    Bxh�  "          @�@@  �l(�@�{BHp�C���@@  ��(�@r�\A�  C��                                    Bxh`  
Z          @�(�@N{�H��@��BQ{C��)@N{���@�(�B�\C��)                                    Bxh"  �          @��@���AG�@��Bjp�C�33@�����@�(�BG�C�E                                    Bxh0�  �          @�@XQ����@�{Bc��C�xR@XQ���@�33B${C��                                    Bxh?R  �          @��@�
�\)@�G�Bz(�C�g�@�
���H@��B-Q�C�                                    BxhM�  
�          @��
@L��?��@�z�Bf�A(��@L�Ϳ�Q�@��RB[{C�ٚ                                    Bxh\�  
�          @��@���>��R@�G�B@{@~{@��ÿ�z�@���B2��C��=                                    BxhkD  T          @�@�\)@���@%�A��
B/33@�\)@Z=q@�
=B�B	��                                    Bxhy�  �          @�33@���@�=q?333@��
BZ��@���@��\@?\)A�z�BLQ�                                    Bxh��  
�          @��R@�ff@�@z�Ay�BT��@�ff@�@�ffB33B9��                                    Bxh�6  
�          @�@���@�=q?\A;�B9��@���@�=q@\��A�\)B"(�                                    Bxh��  
�          @�p�@�p�@��H?�(�A7\)B=p�@�p�@��
@Z=qA�33B&ff                                    Bxh��  
�          @�\)@���@�=q@��A��B\)@���@J�H@hQ�A��RA�(�                                    Bxh�(  
�          @���@��@<��@n{A�Q�A���@��?�ff@�=qB!  A_
=                                    Bxh��  "          @�  @���?�p�@��
BA�RA�
=@��׿.{@���BI�C���                                    Bxh�t  
�          @�R@���?�@�33B:�A�  @��׾���@�33BF��C�u�                                    Bxh�  "          @�\@���8Q�@���B&p�C��@����p�@��HB��C��q                                    Bxh��  T          @�\)@�33@C�
@�A�{B�\@�33?�z�@Z=qB�\A�                                    Bxhf  �          @�@��@�ff�G����B#  @��@��
?�33A
�RB!Q�                                    Bxh  	�          @�33@�=q@�녿333���B>�\@�=q@�p�?��HA1B;p�                                    Bxh)�  �          @��@��R@��
?(�@���BH  @��R@�
=@,(�A�\)B:{                                    Bxh8X  "          @�@��@�p�?O\)@ə�B=�@��@��R@333A�\)B-�\                                    BxhF�  "          @�ff@~�R@��H?�Q�A(�Bcp�@~�R@�ff@X��A��BRQ�                                    BxhU�  �          @�\@aG�@���?\)@�ffBr@aG�@��@8Q�A�Bgff                                    BxhdJ  T          @�@>{@У�=�Q�?0��B���@>{@�Q�@!G�A���B~                                    Bxhr�  �          @�\@'
=@�(��33���B��=@'
=@���>�  @   B��                                    Bxh��  T          @�p�@j�H@�\)���
=Bn{@j�H@�
=?��RAQ�Bm�
                                    Bxh�<  T          @��@W
=@��?&ff@��Bq�@W
=@��@3�
A�Q�Be\)                                    Bxh��  
�          @�Q�@Fff@Ǯ����
ffB~\)@Fff@�?��A/
=B}�\                                    Bxh��  
�          @�
=@%�@�{� ����  B��R@%�@����uB��q                                    Bxh�.  
(          @�33@9��@ƸR��
���HB�
=@9��@�(�>#�
?��\B���                                    Bxh��  
�          @�(�@l��@��H�(�����Bn�@l��@�z�?޸RAY�Bk�\                                    Bxh�z  
1          @�(�@g�@�(������Q�Bq�R@g�@�(�?���Ah��Bm                                    Bxh�   �          @�@Vff@Ӆ���
� ��B|�\@Vff@ȣ�@��A��
Bw�                                    Bxh��  T          @�ff@=p�@�{��p����B�B�@=p�@�?���A#�
B�#�                                    Bxhl  "          @�
=@*=q@أ׿У��I��B�\@*=q@���?xQ�@�Q�B�                                    Bxh  
�          @�R@J=q@��ÿ\�=�B�ff@J=q@�(�?z�H@�B��                                    Bxh"�  
�          @�p�@Q�@�\)�����t  B�{@Q�@�\)?&ff@���B�\)                                    Bxh1^  ]          @�ff@>�R@�z��G��;\)B��R@>�R@�\)?��\@���B�L�                                    Bxh@  "          @�
=@J�H@�=q�Ǯ�@��B��=@J�H@�?p��@��B�Q�                                    BxhN�  
�          @�  @@  @�
=������HB�33@@  @�33>u?�\)B���                                    Bxh]P  
�          @�ff@1G�@Å�N�R��{B�B�@1G�@ۅ�L����z�B��                                    Bxhk�  
�          @�p�@8��@�{�
�H����B���@8��@�G�>��R@��B��
                                    Bxhz�  "          @�(�@G�@���$z���
=B|z�@G�@�p��#�
���\B���                                    Bxh�B  	�          @�p�@0��@θR����B��@0��@��
>��?�33B�.                                    Bxh��  �          @�p�@�Q�@�p��#�
��p�BG�@�Q�@��H?��AmG�BAG�                                    Bxh��  �          @�ff@|��@ə�>���@"�\Bg��@|��@�Q�@%�A��B^\)                                    Bxh�4  
�          @�
=@g
=@�\)>���@#33Bsp�@g
=@�@(Q�A�=qBj��                                    Bxh��  	�          @�G�@dz�@�G��333��33Bu��@dz�@�33?�Q�AQ�Br��                                    BxhҀ  T          @��H@��\@ƸR?���A!p�Bb�H@��\@��\@]p�A�{BRQ�                                    Bxh�&  T          @���@r�\@�?fff@���Bn�@r�\@�{@H��AŅBaff                                    Bxh��  
Z          @�33@b�\@�(�?�ABwff@b�\@���@]p�A�(�Bi��                                    Bxh�r  
�          @�z�@0  @�Q�?��HAp(�B�Ǯ@0  @��
@�\)BG�B�
                                    Bxh  
�          @��H@Tz�@�G�?��AV�RBt@Tz�@��@k�A�Q�Bb�                                    Bxh�  �          @��H@X��@���?�p�A@z�Br��@X��@�(�@aG�A�z�BbG�                                    Bxh*d  �          @�Q�@hQ�@��R?0��@���Bj��@hQ�@��\@0  A��HB_
=                                    Bxh9
  �          @�p�@z�H@��R>\@EBb��@z�H@�{@p�A�G�BX�
                                    BxhG�  
�          @�@�  @�\)���
�+�BP�H@�  @��?���Ax  BJ�                                    BxhVV  
�          @��@e�@��R@���B\)BL��@e�@%@��BR�
BQ�                                    Bxhd�  
�          @�@�(�@�ff@S�
A�{B<33@�(�@O\)@�  B$Q�B�                                    Bxhs�  T          @�G�@�  @���@<��A��B=�@�  @c�
@�\)B��BG�                                    Bxh�H  �          @�\)@�{@���@Y��A���BC�@�{@S�
@��B)ffB\)                                    Bxh��  �          @ڏ\@��@�(�?\)@�=qB&�@��@z=q@33A�z�B(�                                    Bxh��  
Z          @�\)@dz�@����R�N�A�{@dz�@p��������RB:                                      Bxh�:  T          @ۅ@r�\?�Q���ff�R��A���@r�\@^�R�����%33B*Q�                                    Bxh��  T          @��H@���@:�H�����z�B=q@���@��8Q��ȏ\B*                                    Bxhˆ  T          @�Q�@��@����$z���
=B�@��@�G��z�H� ��B/G�                                    Bxh�,  �          @�=q@�ff@W
=�>�R���
A�\)@�ff@�{����V�RB{                                    Bxh��  
�          @�p�@���@Z=q�.�R��33B�@���@�z῱��:ffB�H                                    Bxh�x  �          @�ff@��@^{�33����B {@��@����xQ�� Q�Bz�                                    Bxh  T          @��@���@G
=�\)��Q�A�R@���@q녿��\�'
=B�
                                    Bxh�  	�          @�  @�\)@c�
�   ���B  @�\)@�ff������B�H                                    Bxh#j  �          @��@�@c�
�(����HB =q@�@��H�W
=���
B{                                    Bxh2  T          @�=q@���@9���333����A�G�@���@k���33�X(�B
=                                    Bxh@�  
�          @�\@�ff@@���3�
���\A�
=@�ff@r�\��\)�T��B�\                                    BxhO\  �          @���@`  @�=q�k��BRG�@`  @��?.{@׮BSG�                                    Bxh^  �          @��
@l(�@�      <��
Be\)@l(�@��?���A���B^�                                    Bxhl�  �          @���@��H@�
=�����=qBT�R@��H@��\?��A3�
BQ�H                                    Bxh{N  
(          @��
@z=q@�=q�W
=��=qB[�R@z=q@���?��A�BZ�H                                    Bxh��  
�          @��
@�p�@��ÿ���-�BN�H@�p�@��?\)@��RBQ�                                    Bxh��  �          @�z�@��@����
=�?�BL
=@��@��>��@\(�BO�R                                    Bxh�@  
�          @�ff@�Q�@�33��=q�1p�BA�H@�Q�@�  >�ff@p  BE=q                                    Bxh��  "          @�@��R@�p���(��"�RB9
=@��R@���?   @��B;�                                    BxhČ  
�          @�{@�(�@����Q���\B<z�@�(�@�33?��@�G�B?                                      Bxh�2  �          @�p�@���@��Ϳ����=qBQ�H@���@��R?E�@���BS
=                                    Bxh��  �          @�@y��@�녿����B[�H@y��@�(�?=p�@�ffB]G�                                    Bxh�~  K          @���@w�@�G���z��<z�B\�@w�@�{?�\@��B_{                                    Bxh�$  
�          @�{@n{@��R��  �&=qBc�@n{@���?5@�p�BeQ�                                    Bxh �  
�          @��@��\@��Ϳc�
��G�BX��@��\@��
?��
A�\BX(�                                    Bxh p  �          @ᙚ@�  @�{�����HB[�@�  @��?Y��@��B\p�                                    Bxh +  T          @�\@n{@�{��  �=qBg@n{@�?�G�A\)Bg                                    Bxh 9�  T          @�=q@O\)@�33�����4��Bx�@O\)@ƸR?333@�\)By�
                                    Bxh Hb  
�          @�z�@s�
@�(�����'
=Bd33@s�
@�\)?5@�
=Be�                                    Bxh W  �          @��@"�\@��>\@W�B�@"�\@���@z�A�
=B��                                    Bxh e�  "          @�?^�R@\@Dz�Aԏ\B�?^�R@��@�z�B5��B�G�                                    Bxh tT  "          @ۅ@0��@�@(��A��B�33@0��@�Q�@�33BBj(�                                    Bxh ��  "          @ۅ?��R@Å@.{A��HB���?��R@�z�@��B&B�(�                                    Bxh ��  
�          @��?���@�=q@,(�A�z�B�#�?���@��
@���B&�B��
                                    Bxh �F  "          @���?�{@��@!�A�p�B��?�{@�33@��\BG�B�
=                                    Bxh ��  T          @أ�@   @�  ?�(�Al��B��R@   @��
@c�
B {B�
                                    Bxh ��  �          @���@A�@��@p�A�=qBu�@A�@�G�@x��B��Ba\)                                    Bxh �8  
�          @ᙚ@�G�@�(�@W
=A��
B6�@�G�@Dz�@��HB$=qBG�                                    Bxh ��  
Z          @ᙚ@��H@�  @L��A�(�BE(�@��H@^{@���B!Q�B!\)                                    Bxh �  �          @�
=@`��@�=q@O\)Aݙ�B]Q�@`��@qG�@�{B)p�B<�                                    Bxh �*  
�          @��@��@�p�@<(�Aə�B4=q@��@O\)@�{B�HBp�                                    Bxh!�  
�          @��@{�@���@C�
A��
BJ��@{�@dz�@�BG�B(��                                    Bxh!v  �          @�
=@u�@�G�@R�\A�\)BMQ�@u�@_\)@�z�B'Q�B)\)                                    Bxh!$  "          @�p�@j�H@�\)@G
=A�\)BV=q@j�H@n�R@���B#ffB5�R                                    Bxh!2�  
�          @�33@s33@��\@Tz�A���BIG�@s33@R�\@�33B)��B#��                                    Bxh!Ah  "          @�@C33@�=q@P  A߅Bp��@C33@�Q�@���B-��BS
=                                    Bxh!P  "          @�p�@!�@���@N{A�z�B��H@!�@�  @���B0�RBl��                                    Bxh!^�  �          @�ff?�(�@�=q@��HBp�B�� ?�(�@R�\@���BZz�Bl�                                    Bxh!mZ  �          @���@   @�z�@��B933B�u�@   @=q@��Bw�\BI��                                    Bxh!|   "          @��
@�@w
=@��
B>Br�@�@��@��HBy�\B1�\                                    Bxh!��  �          @�p�@�@�=q@��B'Q�Bu�@�@.{@���BcBC�H                                    Bxh!�L  "          @�Q�@*=q@B�\@�33BS�
BC{@*=q?��H@���B�G�A��
                                    Bxh!��  T          @ٙ�@(Q�@L(�@��HBP�
BI��@(Q�?�\)@�=qBAۮ                                    Bxh!��  "          @�Q�?��@J=q@�  Bk�B�z�?��?�(�@�ffB�  B$z�                                    Bxh!�>  �          @ڏ\?�@�@\B�(�BP?�>���@�  B�aHA$��                                    Bxh!��  T          @�\)?�Q�@$z�@�ffBz�Bb?�Q�?��@θRB��fA��R                                    Bxh!�  
Z          @�(�@�@E@��
BR  BO�@�?�{@��\B�  A�p�                                    Bxh!�0  �          @Ӆ@@��@p��@�  B,{BMp�@@��@(�@�{B^z�B=q                                    Bxh!��  T          @�(�@0  @g
=@��HB;ffBR��@0  ?�Q�@�
=Bn  B                                    Bxh"|  �          @Ӆ@��@Z�H@�
=BP\)Bd��@��?�33@���B��fB\)                                    Bxh""  �          @Ӆ?��
@a�@�z�BQ��B|��?��
?��@�\)B�33B4G�                                    Bxh"+�  �          @�?�  @C33@���Bq�\B�L�?�  ?���@θRB��qB'��                                    Bxh":n  "          @��?��@�@���B��qBy�\?��>k�@�B�z�A4��                                    Bxh"I  
(          @���?�(�?��R@\B�=qBD�H?�(�=#�
@���B��?�33                                    Bxh"W�  "          @�ff?�
=?��@�=qB���BC�H?�
=��Q�@У�B��C��
                                    Bxh"f`  �          @�33?�Q�?��@���B���B0p�?�Q�z�@���B���C��                                    Bxh"u  T          @�33?���@{@��B�33BZ��?���>���@�(�B��)A+33                                    Bxh"��  �          @���@�@R�\@�{BZQ�Bip�@�?��R@�B���Bz�                                    Bxh"�R  �          @ə�?���@1G�@�33Bi{Bb�
?���?��\@�{B�Q�A���                                    Bxh"��  �          @�  ?�\)@�@��B��BM�?�\)>�{@�B�u�A=�                                    Bxh"��  �          @���?�{@33@��RBy�Bv33?�{?�R@��
B��qA�=q                                    Bxh"�D  T          @��\?�=q?�Q�@��
B�#�B=�R?�=q���
@��\B��C�7
                                    Bxh"��  T          @�33?�ff?��@���B��RBP
=?�ff��\)@��RB��HC��                                    Bxh"ې  T          @���?�ff?ٙ�@�=qB�ffBiG�?�ff>���@�(�B�G�A�z�                                    Bxh"�6  "          @�\)?^�R?�@�\)B}��B�� ?^�R?z�@��B��\B�                                    Bxh"��  �          @���?��H@G�@��Br�BlG�?��H?0��@�=qB�(�A�p�                                    Bxh#�  "          @���>�(�?�ff@b�\ByB�#�>�(�?+�@z�HB��Bd�
                                    Bxh#(  
(          @}p��
=?���@l(�B�ǮB�#׿
=�#�
@vffB��\C4��                                    Bxh#$�  	�          @z�H�@  ?c�
@j=qB��3CY��@  �k�@p��B�G�CE.                                    Bxh#3t  
�          @��R��\)?�{@~{B��HB��
��\)��G�@��
B�� CI��                                    Bxh#B  
�          @��H�(��?���@��\B�
=B�녿(�þ��@�
=B�ǮC@��                                    Bxh#P�  �          @�G�?\@'
=@G
=B7=qBo��?\?˅@o\)Bn\)B9G�                                    Bxh#_f  �          @fff=�Q�?�G�@8��Bj�B�W
=�Q�?Tz�@Q�B�Q�B��                                    Bxh#n  
�          @�(���=q>u@z�HB��\C'�׿�=q�k�@tz�B�G�C\}q                                    Bxh#|�  �          @�{��ff?Tz�@{�B�aHCٚ��ff��Q�@�  B���CF��                                    Bxh#�X  
�          @�(���=q?�p�@�=qB�B�B��쾊=q>u@�=qB�p�C�                                    Bxh#��  
Z          @��<�@%@q�B^G�B��3<�?���@��B��B��=                                    Bxh#��  �          @�������@+�@hQ�BV33B�\����?�G�@�  B�B�33                                    Bxh#�J  �          @�>W
=@�H@l(�Bb�B��>W
=?�  @�\)B���B���                                    Bxh#��  T          @|��?@  ?�Q�@\(�Bw�B�33?@  ?�R@q�B��B=q                                    Bxh#Ԗ  T          @��
>u@G�@^{Bn��B�� >u?n{@z=qB�G�B�G�                                    Bxh#�<  �          @Y������?��@2�\B_ffB��
����?+�@G
=B�\CaH                                    Bxh#��  �          @^�R?xQ�?�\)@9��Bc��Blp�?xQ�?8Q�@O\)B�.Bff                                    Bxh$ �  T          @���?��@��@K�BH�
B���?��?���@n�RB��BI=q                                    Bxh$.  
�          @��?��@+�@c�
BFBy��?��?�ff@�p�B}�HB@
=                                    Bxh$�  	�          @���@   @*�H@{�BFz�BT�\@   ?�Q�@���Bu�\B                                      Bxh$,z  �          @��?��@c33@��HB5��Bw�H?��@\)@�Bm�BG\)                                    Bxh$;   
�          @�{?��R@j�H@�
=B5ffBu��?��R@z�@��HBl{BEG�                                    Bxh$I�  	.          @��R@�@^{@��B<
=Bi��@�@@��Bp{B2{                                    Bxh$Xl  T          @��@{@]p�@�33B:p�Be
=@{@@���Bmp�B,��                                    Bxh$g  T          @�ff@�@Z�H@�33B<{Bf
=@�@33@�z�Bo�B-G�                                    Bxh$u�  
�          @��@ff@c33@�33B:(�Bm�\@ff@�@�p�Bn�B8\)                                    Bxh$�^  
�          @�G�@G�@dz�@�B<BrG�@G�@(�@�  Br  B==q                                    Bxh$�  
�          @��H?��R@mp�@���B;��B�ff?��R@�@���Bv(�Bgz�                                    Bxh$��  
(          @��H?�@L(�@��HBL��B��?�?�@��B���BR                                      Bxh$�P  "          @���?�33@G�@���Bm�Bh�H?�33?h��@�
=B�ǮB=q                                    Bxh$��  �          @�(�?˅@ff@���Bh(�B_\)?˅?}p�@���B�
=A�z�                                    Bxh$͜            @�{?��H@
=@���BiQ�BL33?��H?O\)@��B�.A�                                      Bxh$�B  	�          @���?�{?�(�@uB�\B(  ?�{=�G�@���B���@��                                    Bxh$��  �          @���@z�?�33@�
=Bg�Bff@z�>�
=@���B���A7�                                    Bxh$��  L          @�  @
=?޸R@��\B_��B@
=>�@�z�BzffA733                                    Bxh%4  
b          @��?�{��@��B��C�Q�?�{��p�@���B���C�Z�                                    Bxh%�  T          @��@33?���@�  Bxz�B�
@33=�Q�@��RB�
=@=q                                    Bxh%%�  T          @���@33@{@�  BZG�B0
=@33?fff@�{B}�\A�p�                                    Bxh%4&  "          @�
=@/\)@fff@�
=B>��BS�@/\)@ff@�Q�BlG�B=q                                    Bxh%B�  �          @�@7
=@Fff@�BK�B=
=@7
=?Ǯ@��\Bs=qA���                                    Bxh%Qr  
�          @���@*=q@p��@���B=��BZ��@*=q@\)@�33Bl�B p�                                    Bxh%`  "          @�(�@.{@|(�@�ffB7��B]z�@.{@(�@��\Bg�B'�                                    Bxh%n�  
�          @�G�@,��@x��@��B7
=B]
=@,��@�H@��BfB'                                    Bxh%}d  �          @ָR@��@s�
@�{B=Be{@��@�@���BnB.G�                                    Bxh%�
  "          @�{@�@�(�@�\)B4=qBtz�@�@,��@�p�BhQ�BG=q                                    Bxh%��  "          @�
=@��@�G�@�B%  B�ff@��@K�@��B[ffB]p�                                    Bxh%�V  T          @��
@!�@p��@���BB�B`(�@!�@�R@�
=Br
=B%��                                    Bxh%��  "          @��
@*=q@���@�(�B5\)Bb(�@*=q@#�
@�G�BeB/��                                    Bxh%Ƣ  T          @�=q@ ��@~�R@�p�B9�Bf�H@ ��@ ��@���Bj33B3�                                    Bxh%�H            @أ�@��@��@�G�B433Bl�H@��@*�H@��RBfp�B>33                                    Bxh%��  
�          @�z�@3�
@��@��B{Bi�\@3�
@P  @��
BMffBD�\                                    Bxh%�  
�          @��@1G�@���@���B)��Bd�@1G�@8Q�@�(�BZz�B8��                                    Bxh&:  "          @�z�@8Q�@�
=@��
B��Bd�@8Q�@I��@���BN��B>33                                    Bxh&�  
�          @��
@J=q@��@aG�B �B^{@J=q@`��@��B1G�B?��                                    Bxh&�  "          @��@J�H@��H@\(�A��HB]p�@J�H@aG�@���B.�HB?��                                    Bxh&-,  T          @˅@QG�@�{@J�HA��HBV\)@QG�@\(�@�\)B&�B9                                    Bxh&;�  "          @ȣ�@K�@��@1G�A�z�B]�@K�@n{@x��B�BE��                                    Bxh&Jx  "          @˅@<(�@�Q�@XQ�B z�Bc��@<(�@]p�@�ffB1�RBF�
                                    Bxh&Y  
�          @�33@5�@���@Z=qBBh�\@5�@`  @�\)B3�BL
=                                    Bxh&g�  
�          @�=q@)��@��@j=qB��Bl
=@)��@R�\@�{B?��BL                                    Bxh&vj  �          @��@%@�Q�@���B��Bj��@%@C�
@�  BM�BF�                                    Bxh&�  T          @�z�@/\)@��R@U�A��BoG�@/\)@k�@�ffB0�BU33                                    Bxh&��  �          @���@%@���@R�\A�p�Bv(�@%@p��@�B0p�B]��                                    Bxh&�\  T          @�  ?��\@�G�?�A�G�B��?��\@���@C�
B��B��R                                    Bxh&�  T          @�p�?���@�Q�@,��A�(�B�W
?���@j=q@q�B)z�Bw�H                                    Bxh&��  
�          @��R@(�@���@K�B��Bk@(�@E@��
B9G�BN                                    Bxh&�N  T          @��\?�Q�@l(�@c�
B'��B�p�?�Q�@(��@�z�B^�\Bu��                                    Bxh&��  
�          @��?�=q@p��@h��B)\)B�{?�=q@,��@��B`�B~��                                    Bxh&�  
�          @�33?Y��@fff@w�B8��B�W
?Y��@\)@�p�Brz�B�\                                    Bxh&�@  
b          @��R@%�@�  @{A�{Bk  @%�@b�\@O\)B�
BW��                                    Bxh'�  
�          @�ff?��@y��@~{B0{B�W
?��@0  @��HBg�\B���                                    Bxh'�  
Z          @�33?��@r�\@�G�B>{B�\)?��@#�
@��
Bvz�B��H                                    Bxh'&2  �          @���?n{@S33@��BM�B�k�?n{@@���B��B��
                                    Bxh'4�  "          @�33>�(�@Q�@���BW�HB�.>�(�@ ��@��B�#�B�
=                                    Bxh'C~  T          @�G�?!G�@���@�z�B;�
B�33?!G�@333@���Bu�HB�u�                                    Bxh'R$  T          @�{?�Q�@p��@`��B�\B{  ?�Q�@0  @�33BQ(�B[33                                    Bxh'`�  
�          @�ff@S�
@j�H@5�A��B?�
@S�
@5�@j�HB �\B"�                                    Bxh'op  
�          @�  @E�@�(�@   A�33BUQ�@E�@W
=@^{B
=B>33                                    Bxh'~  "          @ə�@/\)@�Q�@%�A�ffBu�@/\)@�{@p��B�\Bc(�                                    Bxh'��  ~          @�G�@G�@�
=@I��A��B��R@G�@q�@��B2�
Bwz�                                    Bxh'�b  \          @θR@ ��@���@P��A��B�G�@ ��@��@�ffB/�\B�\)                                    Bxh'�  
�          @�{@{@�
=@2�\A�Q�B�k�@{@�33@�Q�B  Bq��                                    Bxh'��  �          @��@�
@�@X��A�\)B�@�
@z�H@���B3=qBm�                                    Bxh'�T  
�          @�(�?�33@�  @�  B�B�.?�33@fff@��BR�B��                                     Bxh'��  
�          @�\)?�p�@��
@w
=Bp�B��q?�p�@\)@���BA��B~�                                    Bxh'�  T          @�  ?��@�p�@z=qB��B��?��@���@�=qBC�B�{                                    Bxh'�F  
�          @�{?\@�G�@Z=qA�\)B�aH?\@���@�B1�B��H                                    Bxh(�  
�          @�{?�ff@��R@~�RB(�B��?�ff@���@���BI��B�ff                                    Bxh(�  
�          @ָR?��@�33@\)BG�B�(�?��@�{@�{BKQ�B���                                    Bxh(8  
�          @�(�>�Q�@��
@�{B��B��H>�Q�@z�H@��\BV
=B�k�                                    Bxh(-�  
�          @���>��@�ff@��
B%G�B�� >��@n{@��RB^Q�B�aH                                    Bxh(<�  �          @�?��@�p�@mp�Bz�B�Q�?��@��H@�{B?Q�B��)                                    Bxh(K*  �          @�  ?�p�@�=q@aG�BffB���?�p�@���@�\)B:�RB���                                    Bxh(Y�  
�          @�  ?�=q@���@xQ�B��B�ff?�=q@z=q@��BI�\B��=                                    Bxh(hv  
�          @У�?�z�@���@���BB�Ǯ?�z�@p  @�(�BO��B��R                                    Bxh(w  T          @Ϯ?�=q@�Q�@s�
B��B�?�=q@z�H@�BEz�B�#�                                    Bxh(��  
�          @�
=?�=q@�  @��\B(��B�L�?�=q@S33@���B]p�B��H                                    Bxh(�h  T          @�ff@33@�@c33Bz�B��R@33@z=q@�z�B9By�                                    Bxh(�  
�          @�?�
=@��R@���BG�B��R?�
=@e�@���BO��B�Ǯ                                    Bxh(��  
�          @�{?���@�=q@N{A�z�B���?���@�(�@�p�B-B��                                    Bxh(�Z  
�          @θR?�G�@��@HQ�A�Q�B���?�G�@�{@��HB)�B��q                                    Bxh(�   T          @�G�?���@�z�@dz�B33B�\)?���@xQ�@���B@
=B��)                                    Bxh(ݦ  
�          @�=q?�(�@�ff@<��A�z�B��?�(�@��\@�B%�B�
=                                    Bxh(�L  
�          @���?�p�@��
@A�A�  B�=q?�p�@�
=@��B%��B��3                                    Bxh(��  
�          @�  ?��R@�33@9��Aԏ\B�(�?��R@�\)@��B�B�                                    Bxh)	�  T          @Ϯ?�(�@�p�@VffA���B��3?�(�@��R@��B2p�B�p�                                    Bxh)>  T          @�G�?�33@��R@XQ�A�p�B�.?�33@��@��HB2��B�33                                    Bxh)&�  
�          @�33?n{@��\@q�B��B�L�?n{@���@�ffBB\)B��                                    Bxh)5�  
�          @�?xQ�@��\@y��B�B�k�?xQ�@��@�=qBEz�B�G�                                    Bxh)D0  �          @�ff?&ff@�p�@�{B=qB�?&ff@���@��BR{B��R                                    Bxh)R�  T          @�?\)@�ff@��B�B��3?\)@��@���BP��B�G�                                    Bxh)a|  �          @���?�@�(�@�ffB�\B�L�?�@~�R@���BSffB���                                    Bxh)p"  T          @�z��G�@@��@��Ba\)B���G�?�
=@���B��CxR                                    Bxh)~�  �          @ə��qG�@$z�@��B.��C���qG�?��H@�33BH�C��                                    Bxh)�n  
Z          @�����?G�@��B3�C)��녾�\)@�p�B6�C7�3                                    Bxh)�  
�          @���~{�=p�@��
BC�\C>���~{����@�=qB2��CL��                                    Bxh)��  
�          @Å�c�
���@��HBBCO���c�
�:�H@�Q�B$=qC[Q�                                    Bxh)�`  
�          @�Q��;�>L��@�p�Ba��C0+��;��Y��@�33B[�\CD:�                                    Bxh)�  
�          @���
=@Mp�@�p�BJ{B����
=@G�@���BtQ�CB�                                    Bxh)֬  
�          @�
=�Q�>W
=@�  Br��C.�)�Q�J=q@�Bl(�CFff                                    Bxh)�R  
Z          @�  �*�H�
=@���BU=qC]n�*�H�Z=q@��HB-ffCg��                                    Bxh)��  �          @�=q�\(��p�@��\B/�CW��\(��Vff@X��B��C`0�                                    Bxh*�  �          @Å��=q��{@n{B�HCD����=q���@S33B33CM��                                    Bxh*D  
(          @�z���(�?�@��RB.p�C,����(���(�@�
=B/=qC9�f                                    Bxh*�  	�          @ƸR�Q�?z�@�Be
=C*��Q녿&ff@�p�Bdp�C?+�                                    Bxh*.�  
�          @�\)�c33@��@���B>Q�C޸�c33?��
@��BT33C#�)                                    Bxh*=6  
/          @�ff��@#33@���B_�
Cٚ��?��
@�33B��C��                                    Bxh*K�  
�          @��\)?�  @���Bp�C�3�\)>��@��B�\C)ff                                    Bxh*Z�  
�          @�(���
@�
@�ffBm{C
E��
?L��@�G�B�k�C �q                                    Bxh*i(  �          @�� ��@%@�Bf��B��q� ��?���@�z�B�C�                                    Bxh*w�  "          @�p��'
=@8Q�@�Q�BK
=C0��'
=?ٙ�@���Bn(�C��                                    Bxh*�t  "          @�ff��@\(�@��
BOp�B��
��@{@�G�B}G�B��                                    Bxh*�  �          @��Ϳ��
@Z�H@�33BQQ�B�33���
@p�@�Q�B�B�k�                                    Bxh*��  
�          @��
���R@n�R@��BC��Bߨ����R@$z�@��HBs��B�\)                                    Bxh*�f  �          @�G��L��@�
=@�\)B2Q�B�aH�L��@H��@��HBfG�BЅ                                    Bxh*�  
�          @��=�\)@�\)@{�B$�B�{=�\)@]p�@�33BZ
=B��                                    Bxh*ϲ  T          @�\)>��@���@p  B�B�k�>��@b�\@�{BS�\B���                                    Bxh*�X  �          @�ff>�p�@�{@fffB�RB�{>�p�@o\)@��\BJz�B���                                    Bxh*��  "          @�
=?��@�=q@\(�B
=B��?��@z=q@�ffBAffB��                                    Bxh*��  
�          @�ff?���@�{@*=qA�G�B��?���@�{@q�B�B�\                                    Bxh+
J  "          @�?��H@�
=@Q�A��
B��3?��H@�G�@`��BffB�L�                                    Bxh+�  
�          @\?У�@�33@A��B��?У�@�p�@_\)B�HB��\                                    Bxh+'�  
�          @�?�\)@��R@*�HA�{B��=?�\)@�
=@r�\B{B���                                    Bxh+6<  �          @�p��0��@b�\@�33BUG�B�#׿0��@ff@���B�(�B��)                                    Bxh+D�  
�          @�  �^�R@J�H@��Bi
=B��^�R?��@�z�B�8RB�3                                    Bxh+S�  �          @Ǯ�.{@l(�@�\)BS�B�  �.{@{@�{B�� B���                                    Bxh+b.  
Z          @�=q�=p�@��
@�
=BA�RB�G��=p�@<��@���Bt�
B��                                    Bxh+p�  �          @��ÿE�@�=q@��RB)  B�8R�E�@`  @�z�B\=qB��                                    Bxh+z  �          @Ǯ�#�
@�{@o\)BffB��
�#�
@~{@�  BF��B�ff                                    Bxh+�   T          @ȣ׿8Q�@�=q@�
=B)�RB��)�8Q�@`  @�z�B\��B�33                                    Bxh+��  �          @ə���\)@�@�{B'G�B��ᾏ\)@g
=@�z�B[(�B�                                    Bxh+�l  
�          @��
>#�
@�z�@��B��B�B�>#�
@vff@���BR�B�B�                                    Bxh+�  "          @˅?!G�@���@u�B=qB���?!G�@�G�@��BG�\B�(�                                    Bxh+ȸ  "          @�33>u@�33@p  B�B��=>u@�(�@���BD��B�B�                                    Bxh+�^  	�          @ʏ\=#�
@�33@o\)B
=B�#�=#�
@�(�@�G�BDB��                                    Bxh+�  
(          @�zὣ�
@���@g
=B	�RB��׽��
@��\@�ffB=ffB�                                      Bxh+��  "          @�33>�{@�z�@U�A�G�B�>�{@�Q�@�ffB2�B�\)                                    Bxh,P  �          @��H=�G�@��@@��A�B��3=�G�@��@�p�B%Q�B�Q�                                    Bxh,�  �          @��H>aG�@�
=@*=qA�\)B��=>aG�@�\)@w
=B{B��H                                    Bxh, �  �          @�p�>\@��@{�B{B�z�>\@��@��RBJQ�B�aH                                    Bxh,/B  �          @��>��H@��H@`  B�B�u�>��H@�@�33B7��B�Q�                                    Bxh,=�  �          @˅?#�
@�z�@l(�BB��q?#�
@�{@�\)B@�\B��{                                    Bxh,L�  �          @ʏ\?5@�z�@g
=BffB�L�?5@��R@��B>  B��H                                    Bxh,[4  �          @�=q?u@��R@[�B�RB�{?u@�=q@�  B5�
B��f                                    Bxh,i�  "          @�G�?k�@�\)@\��A���B��
?k�@��\@�=qB1�\B�.                                    Bxh,x�  �          @�33?s33@�z�@K�A�33B���?s33@���@�G�B*�RB�aH                                    Bxh,�&  "          @ʏ\@	��@�  @2�\A�p�B��{@	��@�Q�@xQ�BffB��                                    Bxh,��  �          @ȣ�@@��@*�HAɮB�=q@@�ff@o\)B��By
=                                    Bxh,�r  
�          @�\)@z�@�{@#�
A�B��H@z�@�  @hQ�BB{                                      Bxh,�  "          @ƸR@{@���@�HA��B�@{@��@_\)B	{Bt��                                    Bxh,��  "          @�\)@Dz�@�G�?���A�Q�Bj@Dz�@��@@��A�B^�                                    Bxh,�d  "          @�z�@H��@��?��RA`(�Bh@H��@��@#�
A�33B_
=                                    Bxh,�
  
�          @���@`��@�p�?���AT��BT=q@`��@�  @�A�{BI�H                                    Bxh,��  �          @�\)@Z=q@�ff?���AP��BX33@Z=q@���@A��BN(�                                    Bxh,�V  "          @Å@tz�@��>�Q�@Z�HBL�@tz�@���?��AS
=BG�
                                    Bxh-
�  
�          @Å@%�@���?��AH��B��@%�@��@�A�Q�Bx�                                    Bxh-�  
�          @��H?z�H@�33@z�A��\B�#�?z�H@�
=@[�BB�ff                                    Bxh-(H  
�          @�  ?�{@�=q@|(�Bp�B�#�?�{@fff@��HBKG�B�                                    Bxh-6�  
�          @�p�?���@:�H@��BeG�B�Q�?���?�G�@�Q�B�p�BT�\                                    Bxh-E�  �          @��
����?W
=@�G�B�
=B�#׽��;��H@\B�� C�                                    Bxh-T:  �          @�?��@�R@��B�B{�H?��?z�H@�33B�.B#=q                                    Bxh-b�  T          @�z�>#�
@X��@�ffBc�\B�u�>#�
@�@��B���B�ff                                    Bxh-q�  �          @��?��@y��@��BHQ�B�ff?��@/\)@�(�Bw  B���                                    Bxh-�,  T          @�{?�@��@�(�B7��B��
?�@HQ�@�{Bf��B��                                    Bxh-��  "          @�33@@��@�p�B$  B�Q�@@S�
@�  BO�HBf��                                    Bxh-�x  T          @У�?�=q@�33@��
B"��B��?�=q@fff@���BRQ�B�\)                                    Bxh-�  "          @�
=>�z�@�33@�  B$�B�33>�z�@s�
@�ffBV�RB�aH                                    Bxh-��  
�          @��
��@�=q@~�RBQ�B��ͽ�@u@�BP33B��                                     Bxh-�j  
�          @ə���R@��@w
=B33B��H��R@XQ�@�ffBA{B���                                    Bxh-�  �          @�33��{@�=q@��\B0�B�{��{@N�R@���B]�\B��
                                    Bxh-�  
�          @����z�@��@�
=BA=qB����z�@<(�@��Bn33B��H                                    Bxh-�\  
�          @��Ϳ�p�@n�R@��BI��B���p�@"�\@�G�Bs�\B��
                                    Bxh.  
�          @�{��(�@Z�H@��HBP�B�녿�(�@\)@��RBy�\CL�                                    Bxh.�  
�          @Ϯ���
@mp�@��HBO33B�Ǯ���
@!�@���B{�B�Q�                                    Bxh.!N  "          @�(���33@�(�@�33B/��B��H��33@a�@��B_�B�
=                                    Bxh./�  
(          @Ӆ���
@���@��
B1�BӀ ���
@\(�@�\)B`�
B���                                    Bxh.>�  T          @�(����
@�33@�(�B>
=BθR���
@L��@�ffBm��B�                                    Bxh.M@  "          @��Ϳc�
@�(�@�z�B2{B��Ϳc�
@a�@���Bbp�B�\)                                    Bxh.[�  �          @�p��Tz�@�z�@�B?=qBɊ=�Tz�@N�R@�  Bo�\B���                                    Bxh.j�  
�          @�ff�\(�@�p�@�B>�B���\(�@P��@���Bn�RBѨ�                                    Bxh.y2  �          @�G���z�@�\)@���B/�\B��쿴z�@Z=q@��
B^
=B�Ǯ                                    Bxh.��  
�          @�=q��ff@��@�G�B.�HB��)��ff@_\)@��B]�
B��                                    Bxh.�~  "          @��ÿ���@��
@��B6�BՔ{����@QG�@��Be�B��f                                    Bxh.�$  
�          @�Q�333@��@��RB"��B�uÿ333@vff@���BS=qBȳ3                                    Bxh.��  "          @��þ�Q�@���@w
=BB��;�Q�@�=q@���BA�HB��                                     Bxh.�p  �          @��
�}p�@�Q�@�BQ�B�uÿ}p�@�  @���BMffB��f                                    Bxh.�  "          @�z�
=q@��@|��B�HB�k��
=q@�G�@�BI�RB�W
                                    Bxh.߼  �          @�
=�#�
@�Q�@�G�B�B����#�
@���@���BL��B�u�                                    Bxh.�b  �          @�ff���
@�{@���B3{B����
@W�@��Bb�\B���                                    Bxh.�  �          @��H��@|(�@�p�BL=qB�G���@333@���B|B�W
                                    Bxh/�  
�          @�33�B�\@��
@���B?��B�
=�B�\@A�@�p�Bo��B�G�                                    Bxh/T  
�          @�
=����@q�@��BEz�B������@)��@�33Bo�B��q                                    Bxh/(�  �          @��Ϳ�Q�@j=q@�p�BJ�HB���Q�@!�@��HBu�
B�k�                                    Bxh/7�  T          @ʏ\��\)@I��@���B`G�B�uÿ�\)?��H@��B��qC�{                                    Bxh/FF  
�          @�\)��@Dz�@��B_p�B�  ��?�z�@�{B���C#�                                    Bxh/T�  T          @�녿˅@j=q@��\BI�\B�녿˅@#�
@��Bu{B�R                                    Bxh/c�  T          @ƸR��  @N�R@��RBU�B��Ϳ�  @
=@���B~�RC��                                    Bxh/r8  
�          @�  ��
=@a�@�ffBQ�\B�(���
=@=q@��HB}�B�                                    Bxh/��  T          @���\)@ff@�G�Bo�\C���\)?n{@��
B�Q�C\)                                    Bxh/��  
�          @�{�G�@�H@�=qBmffC޸�G�?��R@��RB��fCz�                                    Bxh/�*  
Z          @�  �HQ�@�\@�\)BT\)C  �HQ�?p��@�G�Bi(�C#\)                                    Bxh/��  
�          @�z��#33@%@���BUC}q�#33?�  @��HBt(�C�=                                    Bxh/�v  
�          @�p��@��@>{@��RB:33Ck��@��?��H@��BY  C�R                                    Bxh/�  �          @����\)@B�\@��RBH�C@ �\)?�p�@�  Bk33Cz�                                    Bxh/��  
�          @�p��'�@C�
@��\BB��C���'�@G�@��
BdffCW
                                    Bxh/�h  
�          @���U�@6ff@�G�B1�\Cu��U�?��@���BM�RC��                                    Bxh/�  M          @�{�QG�@-p�@��B:\)CE�QG�?��H@��RBU�\Ck�                                    Bxh0�  
/          @��R�\@%�@���B<��C�)�R�\?���@��RBV�RCu�                                    Bxh0Z  M          @�ff�\(�?�(�@���BI{C8R�\(�?k�@��\B[��C%�                                    Bxh0"   )          @Ǯ�^�R@	��@��RBD
=C8R�^�R?�{@��BX�C"@                                     Bxh00�  	�          @��
�@[�@�p�BK�B��@z�@���Br\)C�                                    Bxh0?L  "          @ʏ\��33@�
=@��B,
=B�uÿ�33@Mp�@��BWQ�B��                                    Bxh0M�  �          @˅�{@y��@�  B4z�B�W
�{@7
=@�
=B]{B���                                    Bxh0\�  "          @ȣ׿���@~�R@���B333B�G�����@=p�@�z�B]�B�Ǯ                                    Bxh0k>  �          @�Q�B�\@�  @~{B
=B�33�B�\@s�
@�(�BM�Bʙ�                                    Bxh0y�  
�          @θR���@���@��B \)B��ÿ��@z�H@�33BP�B�k�                                    Bxh0��  
�          @����\)@�(�@�Q�B'33B�#׿�\)@hQ�@�z�BV{B�8R                                    Bxh0�0  T          @�(���(�@���@�B#��B����(�@[�@�Q�BO  B�                                    Bxh0��  "          @��
����@�z�@|��B=qB�Q����@|��@�z�BK��B��                                    Bxh0�|  
�          @˅�c�
@�(�@z�HB=qBȔ{�c�
@|��@��BH�HB�L�                                    Bxh0�"  
Z          @�ff>k�@��
@uBQ�B�Ǯ>k�@�ff@��\BC�
B���                                    Bxh0��  
Z          @�Q�?Y��@�z�@G�A�G�B��)?Y��@�33@�\)B"��B�#�                                    Bxh0�n  "          @Ӆ?E�@�ff@O\)A��B��=?E�@�z�@��B%��B�                                    Bxh0�  
�          @�z�?n{@�=q@=p�A�
=B�  ?n{@��\@��BQ�B��                                    Bxh0��  S          @�p�>k�@��R@��A�(�B�Q�>k�@�z�@R�\B��B���                                    Bxh1`  "          @ə���@�
=@A�A�G�B�\��@��R@��B#�B��
                                    Bxh1  �          @˅���@�Q�@b�\B��B��þ��@�z�@�=qB7��B��R                                    Bxh1)�  T          @��Ϳ�Q�@�33@`��B
=BӀ ��Q�@�  @�Q�B4p�B�ff                                    Bxh18R  
Z          @˅�J=q@��\@n�RB�
Bř��J=q@�@��RB?�B�L�                                    Bxh1F�  T          @�zῌ��@�G�@Z=qB ��B�p�����@�ff@�{B0Q�BϨ�                                    Bxh1U�  T          @����\@���@y��B&Q�Bνq���\@W
=@��RBU33Bգ�                                    Bxh1dD  T          @����@��@\(�B�B�  ��@���@�(�B=��B���                                    Bxh1r�  T          @�  �0��@��@r�\BB���0��@j=q@�p�BM��B�8R                                    Bxh1��  
�          @Å>�  @��\@��Aə�B�L�>�  @��R@Z=qBG�B��                                    Bxh1�6  "          @��?�Q�@�z�?��
A���B���?�Q�@�z�@;�A�G�B��{                                    Bxh1��  �          @�(�?�@��H?�z�A��HB��?�@��@C33A�RB��3                                    Bxh1��  
�          @�33?�  @�33?�=qA&=qB�Q�?�  @�  @�RA��B�(�                                    Bxh1�(  "          @�(�?��@�\)?p��A+�
B�B�?��@�?�z�A��HB�#�                                    Bxh1��  �          @�\)��
=@E@ ��B��B垸��
=@\)@G
=B=  B��                                    Bxh1�t  
�          @��
�ff����@Q�B%�CQǮ�ff��\)?�=qB

=CY�                                    Bxh1�  "          @��`���G
=@)��A�{C]���`���e?�Q�A���Ca��                                    Bxh1��  "          @��\�,(��p�׾��R�r�\Cjp��,(��g�����YCiW
                                    Bxh2f  	�          @����a��g�?:�HA�\Ca�q�a��l(��L�Ϳ
=CbL�                                    Bxh2  "          @4z�&ff��p�>�z�A>{Cq��&ff��G�<�?�\)Cr�{                                    Bxh2"�  �          @��
@a�@p���ff���RBp�@a�@   =��
?��B(�                                    Bxh21X  T          @�{@@��@n�R���H��(�BLQ�@@��@p  >���@fffBL��                                    Bxh2?�  �          @��;��
@!G�?�=qA�ffB�L;��
@(�?�=qBffB�u�                                    Bxh2N�  
�          @��\�j�H��R@�RA�  CV
�j�H�<��?�33A�  CZ��                                    Bxh2]J  T          @����E�Q�@9��B�CV���E�,(�@Q�A�G�C]                                      Bxh2k�  �          @�z��l(���@=qA�\)CQc��l(��"�\?�A���CV�=                                    Bxh2z�  �          @�(��xQ쿠  @'�B��CE���xQ��\@�\A�=qCL��                                    Bxh2�<  �          @�����Q��@z�A���CIp���Q��z�?�A��CM�                                    Bxh2��  �          @��R��  ���R?�ffA�p�C7� ��  �!G�?���A�Q�C;!H                                    Bxh2��  T          @��\�����G�?�33Ah��CAxR����\?���A6�\CD33                                    Bxh2�.  
�          @��\��33���H?���Aq�C9G���33�J=q?��AYC<aH                                    Bxh2��  �          @�
=���R?�z�?��
Ab{C�����R?˅?�33A��C!O\                                    Bxh2�z  	�          @�p����
@S�
?���AL��C8R���
@>�R?�\)A��
C)                                    Bxh2�   
�          @�
=�tz�@QG�?���AW\)Ck��tz�@<(�?�\)A�\)CaH                                    Bxh2��  �          @��\�H��@dz�?Y��A"�\CW
�H��@S33?˅A��C��                                    Bxh2�l  �          @��
�E@S33?^�RA-p�C{�E@B�\?��A���Ch�                                    Bxh3  T          @�{��H@n{@
=A��B�.��H@L(�@6ffB��B�u�                                    Bxh3�  
�          @��׿�33@�=q?�Q�A�G�B����33@c�
@0��B	p�B�.                                    Bxh3*^  �          @��z�H@�p�?��RAj{B˳3�z�H@���@��Aԣ�Bͮ                                    Bxh39  �          @�=q����@���?��\Ah��B�W
����@���@  AӮB�k�                                    Bxh3G�  "          @�Q��  @��?���A`��B����  @y��@   A�ffB�
=                                    Bxh3VP  "          @��\�Vff@`���(����z�C��Vff@\)����\)C�                                    Bxh3d�  �          @�p��Dz�@:�H�U��Cu��Dz�@c�
�(�����CǮ                                    Bxh3s�  T          @�ff�   ?�33���H�Q�
C���   @.�R�e�0�Cu�                                    Bxh3�B  �          @��
����@��ÿ��H���
B�
=����@�  ��\��  B�z�                                    Bxh3��  T          @�\)����@{���
=���B�R����@�{�O\)� ��B��f                                    Bxh3��  �          @��H��\@�  ?��AUB�3��\@l(�?�\)A�(�B�=q                                    Bxh3�4  
Z          @�z��G�@�����Q�B�8R��G�@�{>��@�p�B�8R                                    Bxh3��  
�          @��?(��@���>��@�z�B�Q�?(��@�=q?�A��\B���                                    Bxh3ˀ  "          @��H?���@�G������T(�B���?���@�p���G����B�ff                                    Bxh3�&  T          @�  ?��H@�  ��H���\B�W
?��H@�p���G����B�ff                                    Bxh3��  "          @�G�?�33?У���  �n  B"��?�33@   �s�
�I�
BS�                                    Bxh3�r  �          @��\?z�H@������mp�B�33?z�H@Fff�e��?(�B��                                    Bxh4  T          @��?Ǯ@$z��<���1Bj��?Ǯ@H�����z�B~
=                                    Bxh4�  �          @n{?�{?����K��l�HB0=q?�{?�(��4z��G  B]=q                                    Bxh4#d  
�          @|��?�33?��L(��\  A��?�33?����7��=��B.�                                    Bxh42
  T          @}p�@L��?����33���A��
@L��?�{��=q��{AՅ                                    Bxh4@�  �          @z�H?���?#�
�(��G�
A���?���?�33�\)�3=qA�(�                                    Bxh4OV  �          @j�H���
�Tz��aG��\Cy:ᾣ�
�u�g�©��C>Ǯ                                    Bxh4]�  �          @g��p�׿�{�R�\.Ce�{�p�׾\�\���CI�
                                    Bxh4l�  "          @c�
��Q쿴z��AG��j�
Ceٚ��Q�8Q��P����CSaH                                    Bxh4{H  "          @e��Q��=q�>{�a�Ch���Q�h���O\)W
CY0�                                    Bxh4��  "          @dz΅{��
=�333�Q{Cg\��{����E�t{CYxR                                    Bxh4��  "          @r�\������Ϳ�������Cgz������(����$CaW
                                    Bxh4�:  �          @h�ÿ��H�.�R�Ǯ��ffCm�3���H���
=�p�Ci�q                                    Bxh4��  T          @^�R����/\)�޸R��Cxn�����
��\�'��Ct��                                    Bxh4Ć  T          @\(��h���)���   ��C{
=�h���	���!G��<�RCw)                                    Bxh4�,  �          @��ÿ�G��333?�{B�CL�Ϳ�G��k�?n{A�=qCS\)                                    Bxh4��  �          @�
=�,��?��@���BgC��,��?5@���B}�C%Q�                                    Bxh4�x  �          @�G�� ��?���@��RBqG�C��� ��?�R@�\)B�� C&�                                    Bxh4�  �          @���   ?��H@��Bz�
C�=�   >W
=@�G�B�(�C/B�                                    Bxh5�  �          @��H���@ff@��
B~ffCB����?8Q�@�B�u�C!�q                                    Bxh5j  T          @���(�@
�H@�G�Bp�C=q�(�?B�\@��
B�W
C �                                    Bxh5+  T          @�����@ff@��HB�� CY����?�R@�z�B�Q�C$T{                                    Bxh59�  T          @����@G�@�G�B�aHC.��?\)@�=qB��HC')                                    Bxh5H\  
�          @����R@��@��
B��C���R?&ff@�B��C#��                                    Bxh5W  �          @�z���
@\)@�\)B�aHC�\��
?8Q�@��B��C ��                                    Bxh5e�  �          @�
=��@  @�G�B�
=CL���?8Q�@��
B�{C!O\                                    Bxh5tN  �          @��
�H@#33@�ffB}�HCT{�
�H?��
@�33B���C�)                                    Bxh5��  �          @�G���Q�@ ��@ڏ\B�.B��\��Q�?u@�
=B��{C��                                    Bxh5��  �          @���ff@�R@أ�B��qCW
�ff?p��@���B�L�C��                                    Bxh5�@  
�          @��Q�@-p�@ָRB{Q�C 33�Q�?�@�z�B�aHC&f                                    Bxh5��  
�          @�=q�
�H@1G�@�ffBy�C 
�
�H?��R@�z�B�k�CW
                                    Bxh5��  
�          @�ff� ��@%�@�B��B��)� ��?�ff@�\B�u�CY�                                    Bxh5�2  T          @�=q��@
=@޸RB���B��H��?E�@��B��C�{                                    Bxh5��  �          @��H���
@+�@��
B�ǮB�{���
?�{@�G�B��C�                                    Bxh5�~  
�          @���@(��@�ffB��RB�����?���@��
B�� C�                                    Bxh5�$  �          @�p��Q�@	��@ϮB�\C�R�Q�?+�@��B��RC"��                                    Bxh6�  T          @�p��
=q@G�@У�B�33C	�
=q?�@��B��qC&:�                                    Bxh6p  
�          @�{���?�33@�(�B��3CG����>��@ڏ\B�33C0                                    Bxh6$  �          @���z�?��@�=qB�z�C���z�aG�@�{B��
C;��                                    Bxh62�  �          @�ff�(�?�(�@ҏ\B�.C�H�(�>aG�@ٙ�B�#�C.E                                    Bxh6Ab  �          @�p����@Q�@�
=B��qC�����?&ff@���B���C#�q                                    Bxh6P  T          @�ff�	��@@θRB  C�{�	��?\(�@�=qB�p�CW
                                    Bxh6^�  T          @�׿�
=@!G�@ӅB��RB��ÿ�
=?}p�@�Q�B�.C33                                    Bxh6mT  "          @�{�p�@
=@�p�B}33C&f�p�?^�R@���B�k�CxR                                    Bxh6{�  �          @�33�@@���Bx�RC���?c�
@�z�B��C8R                                    Bxh6��  �          @陚��\)@5�@�{B|z�B�p���\)?�=q@��B�G�C�
                                    Bxh6�F  
Z          @�33���@&ff@�=qBB�33���?���@߮B�k�C�)                                    Bxh6��  �          @��;�@��@��
BoQ�Cn�;�?G�@ָRB�Q�C%!H                                    Bxh6��  "          @��1�@"�\@�=qBm  C	�{�1�?��@�\)B�W
C�                                    Bxh6�8  �          @���0  @#�
@ȣ�BlQ�C	  �0  ?��@�{B�G�CJ=                                    Bxh6��  �          @�z��*�H@'�@�33Bm�C���*�H?���@أ�B���C\                                    Bxh6�  �          @����8Q�@G�@�
=B^C���8Q�?�33@�Q�B�C33                                    Bxh6�*  �          @�Q��Vff?�(�@�z�Bj��C�\�Vff>�@�p�B|\)C+�=                                    Bxh6��  �          @����l(�?\@��HBgffC�H�l(�=L��@У�Br{C30�                                    Bxh7v  �          @�G��`��?�z�@�{Bn=qC��`�׽�\)@��HBw�
C50�                                    Bxh7  �          @�����
?�\)@\BW��C�=���
>W
=@�G�Bb��C1
                                    Bxh7+�  
�          @�  �n{?�  @�=qBfffC�q�n{=#�
@�  Bp�
C3k�                                    Bxh7:h  "          @�G��7�?�  @׮B}�HC���7�>B�\@޸RB�{C0G�                                    Bxh7I  �          @��?\)@	��@��Br�C=q�?\)?(�@�(�B��C(ff                                    Bxh7W�  4          @��
�aG�@ ��@�ffBgffCO\�aG�>��H@׮Bx�C,�                                    Bxh7fZ  T          @��
�W
=@�@�ffBgffC���W
=?B�\@ٙ�B|�C'Q�                                    Bxh7u   �          @��H�L(�@p�@�p�Bg�CY��L(�?p��@��B�HC#��                                    Bxh7��  �          @��U�@ff@�p�Bfz�C�
�U�?Q�@�G�B|C&�                                    Bxh7�L  �          @���W
=@$z�@�(�Ba��C���W
=?�ff@ٙ�Bzz�C"�H                                    Bxh7��  �          @�33�G
=@4z�@��HBb{C	�\�G
=?�ff@��B~�HC\)                                    Bxh7��  �          @�(��:�H@4z�@�ffBgG�C���:�H?��\@�B���Cp�                                    Bxh7�>            @�z��5�@8Q�@�  Bh�CxR�5�?���@�\)B���C                                      Bxh7��  �          @�(��\)@C33@�G�Bk�
C@ �\)?�(�@��B�z�Cu�                                    Bxh7ۊ  �          @�(���R@Fff@�Q�Bj�C ����R?\@ᙚB�33CxR                                    Bxh7�0  �          @��'
=@H��@�33Be(�C�=�'
=?˅@���B�33C�H                                    Bxh7��  �          @�  �G�@>{@ϮBp=qB��)�G�?�33@�  B�L�Cp�                                    Bxh8|  �          @�\�\)@Tz�@�{Bhz�B���\)?�  @���B�  C�q                                    Bxh8"  �          @��
�z�@W�@�  Bj�B�{�z�?��
@�33B��CG�                                    Bxh8$�  T          @��H���R@g�@�  Bk��B����R@G�@��B�aHB��)                                    Bxh83n  �          @�(����\@{�@θRBgQ�B�#׿��\@�@�{B�
=B�\)                                    Bxh8B  �          @�  �Q�@���@��Bg(�B��
�Q�@�H@�=qB��B�\)                                    Bxh8P�  T          @��ÿ��@|��@�p�Blp�B�G����@�\@��B��\B�\)                                    Bxh8_`  �          @��=�Q�@�33@���BaffB���=�Q�@-p�@�B��\B�33                                    Bxh8n  �          @�
=�aG�@�33@�{Bc�
B��f�aG�@*�H@��B�ǮB���                                    Bxh8|�  �          @��;W
=@���@θRB\
=B�L;W
=@8��@��HB��B�G�                                    Bxh8�R  �          @���z�@���@�  B\G�B�L;�z�@8��@�(�B�\B�k�                                    Bxh8��  �          @��;�(�@�p�@˅BVz�B��\��(�@C33@���B��B��                                    Bxh8��  �          @����z�@��@�{B\Q�B�W
��z�@6ff@��B�(�B��                                     Bxh8�D  T          @�=q��\)@�{@�p�B]33B�.��\)@3�
@���B���B�W
                                    Bxh8��  �          @��ÿ�R@�G�@�ffB`ffB�ff��R@*=q@���B��B�8R                                    Bxh8Ԑ  �          @�녾�p�@�z�@��HBg{B�33��p�@{@�(�B��{B���                                    Bxh8�6  T          @�33���@�p�@�\)B^B�  ���@1G�@��HB���B�aH                                    Bxh8��  �          @��H>�@��@�  B`(�B��>�@,��@�33B�#�B���                                    Bxh9 �  �          @�=q����@�(�@θRB_ffB�������@.�R@��B�
=B�33                                    Bxh9(  �          @�33>�{@�33@У�B`��B�(�>�{@+�@�B��qB��3                                    Bxh9�  �          @��\?z�@��@�B]ffB�
=?z�@0��@陚B�ǮB�G�                                    Bxh9,t  �          @�=q?(�@�\)@�z�B[
=B��?(�@5�@��B���B��R                                    Bxh9;  �          @��H?W
=@�G�@ʏ\BWz�B��?W
=@9��@�
=B��\B��{                                    Bxh9I�  �          @���>�{@���@ӅBc�B��>�{@&ff@�ffB�33B�
=                                    Bxh9Xf  �          @��H>�@�  @�z�BZ�RB�Q�>�@6ff@���B�B���                                    Bxh9g  �          @�z�>��@�  @θRB\(�B��=>��@4z�@�33B��{B�ff                                    Bxh9u�  �          @�=q>aG�@��@���BO��B��
>aG�@L��@��
B��{B�B�                                    Bxh9�X  �          @�(�?\)@�G�@ƸRBP�\B��3?\)@J=q@�B�B���                                    Bxh9��  �          @��?Q�@�p�@�  B]=qB��)?Q�@.�R@��
B��B�\)                                    Bxh9��  �          @��?z�H@�G�@�B^z�B�L�?z�H@'�@���B���B���                                    Bxh9�J  �          @�33>��@��
@�  B_�RB�(�>��@+�@�B��\B�ff                                    Bxh9��  �          @�33?�@��H@�Q�B`z�B�8R?�@(��@�B��HB��=                                    Bxh9͖  �          A ��?5@���@˅BP\)B�� ?5@N{@�33B��qB��                                    Bxh9�<  �          AG�?=p�@�33@ƸRBI{B��{?=p�@\��@�Q�B�(�B��H                                    Bxh9��  �          A=q?��H@���@�\)B>
=B��?��H@j�H@�\BsffB���                                    Bxh9��  �          @��?��
@�  @��HBF��B�33?��
@W�@�(�B|�RB�B�                                    Bxh:.  �          A (�?�(�@�p�@�\)BA�\B�\)?�(�@c�
@��Bw��B�                                    Bxh:�  �          A ��?�{@�Q�@��B@\)B�(�?�{@h��@��HBw�B�{                                    Bxh:%z  �          @��R?\@��R@��\B<Q�B�p�?\@g�@�Bq��B�z�                                    Bxh:4   �          A ��?�=q@�{@��RB>B�8R?�=q@dz�@�G�Bt
=B�ff                                    Bxh:B�  �          AG�?�=q@�(�@�G�BAz�B�Ǯ?�=q@_\)@��
Bv�RB�W
                                    Bxh:Ql  T          A ��?�(�@��@��B6�B�� ?�(�@r�\@�(�BkQ�B�B�                                    Bxh:`  �          A   ?���@�@���B:  B�?���@fff@�z�Bnz�B{=q                                    Bxh:n�  �          @���?���@��@��BEQ�B��{?���@L(�@߮By33Bq(�                                    Bxh:}^  �          @��H?�p�@�\)@�\)B;�\B��{?�p�@Z=q@���Bo33Bo\)                                    Bxh:�  �          @��R@ ��@�
=@��B5ffB��)@ ��@j=q@���Bi�Bt                                    Bxh:��  �          @�
=?�\)@���@��BJ�HB�p�?�\)@Fff@���B�
ByQ�                                    Bxh:�P  �          A ��?���@���@��
BQG�B�?���@:�H@�=qB�\Bu33                                    Bxh:��  �          A z�?޸R@��@ə�BNB�{?޸R@<��@�Q�B�� Bm�
                                    Bxh:Ɯ  �          Ap�?�G�@��@ǮBJp�B�?�G�@G
=@�B
=Bq�                                    Bxh:�B  �          Ap�?�@��R@��
BD��B�33?�@QG�@�p�ByG�Brp�                                    Bxh:��  �          A�?��@�
=@�z�BD33B�Q�?��@Q�@�BxBp
=                                    Bxh:�  �          A=q@Q�@�\)@\BA
=B��@Q�@S33@�(�Bt�\Bd��                                    Bxh;4  �          A�@�
@�{@�33BB�B�@�
@P  @���BvBf�                                    Bxh;�  �          A@�@��@�z�BE
=B��3@�@J=q@�p�Bx��Bb�R                                    Bxh;�  T          A{?��R@�33@ƸRBGffB�aH?��R@HQ�@�B{z�Bf(�                                    Bxh;-&  �          Aff?�  @�
=@θRBR
=B��)?�  @<(�@�{B�.B|(�                                    Bxh;;�  �          A{?��@��H@�=qBX
=B���?��@1�@��B��{B�=q                                    Bxh;Jr  �          A�R?�Q�@��@ڏ\Bc33B�  ?�Q�@(�@�ffB�.B�\                                    Bxh;Y  �          A{?��H@�
=@�p�B\�B�Ǯ?��H@(Q�@�\B�8RB��R                                    Bxh;g�  �          Ap�?���@���@��HBZ�HB��
?���@,��@���B�z�B�ff                                    Bxh;vd  �          A{?O\)@�Q�@ָRB^�
B��?O\)@)��@�z�B�(�B��                                    Bxh;�
  �          AG�?�ff@��\@��BY�RB�33?�ff@0  @�Q�B�=qB�G�                                    Bxh;��  �          A ��?��H@�ff@��BSz�B��?��H@:=q@���B��B��R                                    Bxh;�V  �          Aff?���@��\@���BZ�RB���?���@.{@�33B��RB���                                    Bxh;��  T          Aff?��@���@�  BS�
B��\?��@>{@��B���B�k�                                    Bxh;��  �          Aff?�ff@�ff@�=qBW
=B��?�ff@6ff@��B�.B�                                    Bxh;�H  
�          A   ?�G�@��H@�
=BX  B�.?�G�@1G�@�B��RB��                                    Bxh;��  �          @��H?�ff@���@ə�BV=qB���?�ff@0  @�Q�B���B��                                    Bxh;�  �          @��?�z�@�=q@�
=BS\)B��=?�z�@3�
@�ffB�8RB�#�                                    Bxh;�:  �          @�=q?�\)@�=q@�{BQz�B��3?�\)@3�
@��B��
B�(�                                    Bxh<�  �          @�=q?aG�@�G�@ə�BV��B���?aG�@0  @�Q�B��B�aH                                    Bxh<�  �          @���?�
=@���@�ffBS��B��)?�
=@0��@��B���B���                                    Bxh<&,  �          @���?���@�G�@�  BT��B�aH?���@0��@�
=B�ffB�k�                                    Bxh<4�  �          @�\)?k�@��@ə�BZ�
B�.?k�@$z�@�
=B���B��{                                    Bxh<Cx  �          @��H?�p�@���@�p�BN\)B��=?�p�@5�@���B��)B��
                                    Bxh<R  �          @�  ?��H@��@���BJ  B��?��H@<(�@�G�B��HB�W
                                    Bxh<`�  �          @���?���@�z�@��\BK
=B��?���@<��@��HB�ǮB�=q                                    Bxh<oj  "          @�?h��@�
=@��BH\)B�(�?h��@B�\@���B��
B��                                    Bxh<~  �          @�
=?��@��@�ffBT�B�
=?��@(��@�z�B�G�B�                                    Bxh<��  �          @�>�z�@�G�@�\)Bc��B��f>�z�@  @�=qB�u�B��=                                    Bxh<�\  �          @�R>��
@�
=@�z�B]�\B�aH>��
@��@�G�B�L�B�33                                    Bxh<�  �          @�
=?���@���@�G�BK33B��H?���@5�@���B�ffB�#�                                    Bxh<��  �          @�ff?��
@�{@���B4�RB�Q�?��
@W
=@���Bm=qBx33                                    Bxh<�N  �          @�?�
=@��H@�33BC�\B���?�
=@;�@ӅB{��Bp�                                    Bxh<��  �          @���?��R@�
=@�\)B?�HB�  ?��R@E�@�G�By�\B��                                    Bxh<�  �          @�?�\@��@�Q�B?ffB�u�?�\@AG�@љ�Bw�\Bn�                                    Bxh<�@  �          @��
?�33@�{@�33B:p�B��)?�33@E@��Br{Bip�                                    Bxh=�  �          @�z�@U�@�(�@�{B��BY33@U�@Mp�@�Q�BM  B/�R                                    Bxh=�  �          @�p�@�p�@��@���B�B:Q�@�p�@Fff@���B633Bp�                                    Bxh=2  �          @�{@��\@��R@��\B�B>{@��\@HQ�@��
B9  B�
                                    Bxh=-�  �          @�
=@�G�@�
=@��B��B?�@�G�@G�@�ffB;ffB�\                                    Bxh=<~  �          @�\)@~{@�
=@��Bz�BAp�@~{@E@���B>�B�                                    Bxh=K$  
�          @�\)@|��@�Q�@�
=B  BC{@|��@HQ�@���B>��Bp�                                    Bxh=Y�  �          @�@���@�(�@�{B�B<��@���@@��@��RB<�B�\                                    Bxh=hp  �          @�ff@�Q�@��H@���Bp�B<��@�Q�@<(�@��B@B�H                                    Bxh=w  �          @�  @���@��R@�
=BffB?  @���@Dz�@���B=ffB�R                                    Bxh=��  �          @�{@��
@�33@�(�B��B,�
@��
@0  @��\B7�RB �
                                    Bxh=�b  �          @��H@�{@}p�@�G�B�HB-��@�{@$z�@��RB?�
A�(�                                    Bxh=�  �          @�G�@��@vff@��B�\B"\)@��@"�\@�(�B3ffA��
                                    Bxh=��  �          @��@��@w
=@�
=B�B'ff@��@\)@��B<p�A��
                                    Bxh=�T  �          @�33@���@���@�p�B
=B-  @���@)��@��B;G�A�Q�                                    Bxh=��  �          @�(�@�ff@\)@��\B��B'�\@�ff@(��@�Q�B5�HA�                                    Bxh=ݠ  �          @�p�@�  @z�H@���B��B$G�@�  @#�
@�=qB7
=A�\                                    Bxh=�F  �          @��@�  @{�@�(�B  B$��@�  @$z�@���B6��A홚                                    Bxh=��  �          @�z�@��H@\)@x��A��\B�@��H@0  @�33B${A���                                    Bxh>	�  �          @�Q�@�
=@j=q@��
B
\)B33@�
=@�@�
=B-�AՅ                                    Bxh>8  �          @�=q@�=q@x��@�ffB�HB!��@�=q@$z�@��
B1=qA�
=                                    Bxh>&�  �          @��H@�33@~�R@��BG�B#ff@�33@*�H@��B.=qA�G�                                    Bxh>5�  �          @��@�{@y��@���B�\B�@�{@'�@�ffB*�A��
                                    Bxh>D*  �          @�
=@���@~�R@�{Bz�B+�R@���@)��@�(�B6G�A�{                                    Bxh>R�  �          @���@}p�@u@���B  B0G�@}p�@=q@�{BFG�A�{                                    Bxh>av  �          @��
@�
=@|��@�p�BG�B,p�@�
=@'�@��B7\)A��\                                    Bxh>p  �          @��
@��@|(�@���B	\)B(ff@��@)��@��RB0�
A�=q                                    Bxh>~�  �          @�G�@�Q�@QG�@��Bz�B

=@�Q�?�33@��B4�RA��                                    Bxh>�h  T          @��@�=q@;�@��B�A�ff@�=q?\@�ffB733A�=q                                    Bxh>�  �          @��@�{@N{@��B��B�@�{?�\)@���B.�
A��                                    Bxh>��  �          @��@�
=@X��@tz�A��RB  @�
=@
=q@�(�BffA�{                                    Bxh>�Z  �          @��H@�
=@Q�@i��A��A��@�
=@ff@�B�\A�(�                                    Bxh>�   �          @�@�G�@S�
@_\)A�G�A�z�@�G�@
�H@�G�B{A�G�                                    Bxh>֦  �          @�@�p�@S33@Tz�A���A�@�p�@p�@�(�Bz�A�z�                                    Bxh>�L  �          @���@�
=@K�@[�A�
=A��@�
=@z�@�ffB
\)A���                                    Bxh>��  �          @�(�@��
@^{@R�\A�A�\)@��
@Q�@�z�B�RA�p�                                    Bxh?�  �          @�33@��R@h��@8Q�A�=qB
=@��R@*=q@s33A�
=AǮ                                    Bxh?>  �          @��
@�
=@c33@\)A��A�@�
=@+�@Z=qA�{A�\)                                    Bxh?�  �          @�\@���@Y��@Q�A��
A�p�@���@$z�@P��Aә�A���                                    Bxh?.�  �          @�  @��@b�\@'�A��B �@��@(Q�@a�A�{Aǅ                                    Bxh?=0  �          @�@xQ�@���@��B{BB��@xQ�@AG�@��RB:��B�H                                    Bxh?K�  �          @�\)@u�@��R@�{B��BE\)@u�@C33@���B=  B33                                    Bxh?Z|  �          @�Q�@vff@�{@�\)B  BD�@vff@AG�@��HB>  B(�                                    Bxh?i"  
�          @�=q@_\)@���@���B33BR(�@_\)@A�@��BK  B${                                    Bxh?w�  �          @�p�@J�H@�Q�@�
=B(�B[�@J�H@7
=@�=qB[p�B(=q                                    Bxh?�n  �          @���@G�@��H@���B%�B_  @G�@<��@���BY��B-��                                    Bxh?�  
�          @��
@E@���@�{B({B^�H@E@8Q�@���B\Q�B,{                                    Bxh?��  �          @��
@Fff@�G�@�z�B&�RB^��@Fff@9��@�Q�B[{B,Q�                                    Bxh?�`  �          @�z�@J=q@��H@�33B${B]@J=q@=p�@��BXp�B,ff                                    Bxh?�  �          @�{@`  @�\)@�G�B ��BP  @`  @7
=@���BR33B33                                    Bxh?Ϭ  �          @�@U@��@�Q�B!�BUp�@U@8Q�@�(�BT\)B#(�                                    Bxh?�R  �          @�=q@HQ�@���@��\B%=qB]=q@HQ�@8��@�ffBYB*                                    Bxh?��  �          @��@@��@���@�(�B'�Ba��@@��@8Q�@�Q�B]�B.�
                                    Bxh?��  �          @��H@N{@�33@�{B)��BU@N{@+�@�Q�B\z�B��                                    Bxh@
D  �          @�=q@E@��H@�  B,�
BZ(�@E@)��@�=qB`�\B"z�                                    Bxh@�  T          @陚@1G�@�\)@���B/{Bi{@1G�@1G�@��Bf{B3��                                    Bxh@'�  �          @�@H��@���@��HB.��BWz�@H��@%�@�z�Bb33B��                                    Bxh@66  �          @��@HQ�@��
@�33B-�BY��@HQ�@(��@�p�BaB �                                    Bxh@D�  �          @��@L��@���@�(�B/=qBT�
@L��@"�\@�Bb
=B�
                                    Bxh@S�  T          @�@\(�@��\@�p�B0ffBGz�@\(�@�@��B_��B��                                    Bxh@b(  �          @��
@`  @xQ�@�{B3G�B?�H@`  @Q�@��
B`��A�
=                                    Bxh@p�  
�          @�(�@Z=q@|��@�
=B4
=BD��@Z=q@�@��Bb��B�\                                    Bxh@t  �          @��@J=q@��@�{B4�HBPff@J=q@33@�p�Bf��B(�                                    Bxh@�  �          @�Q�@AG�@�p�@�(�B3BXff@AG�@=q@���Bg�\B��                                    Bxh@��  �          @�{@W�@n{@�(�B6\)B?G�@W�?��R@���Bc�
A�                                      Bxh@�f  T          @��
@O\)@[�@��BAffB:G�@O\)?�33@�33Bl�A�\)                                    Bxh@�  �          @��H@U@^�R@�B<�B8�
@U?޸R@�  Bg�\Aۮ                                    Bxh@Ȳ  �          @�z�@��@,��@�ffB0�A�=q@��?�ff@�G�BL(�AY��                                    Bxh@�X  �          @�G�@��@.�R@�33B.�BQ�@��?�{@�ffBK
=Ag\)                                    Bxh@��            @��@���@4z�@��\B,��B�R@���?���@��RBJ�
AxQ�                                    Bxh@��  �          @�G�@�G�@'�@�ffB'G�A�  @�G�?�ff@���BB{APQ�                                    BxhAJ  T          @�  @��R@#33@�ffB4�RA�
=@��R?c�
@�  BO�A?�                                    BxhA�  �          @�Q�@��
@6ff@�p�B2{B
@��
?�Q�@��BQ��A���                                    BxhA �  T          @�z�@Z�H@>{@�{BC�RB#�R@Z�H?��H@��Bh�A�{                                    BxhA/<  T          @���@^�R@AG�@�z�B@p�B#��@^�R?��\@�=qBe��A��R                                    BxhA=�  
F          @��
@k�@S�
@�Q�B/�\B(
=@k�?�@��BW��A�33                                    BxhAL�  �          @��@i��@\(�@��B-�\B-
=@i��?��@��\BW�\A���                                    BxhA[.            @޸R@g�@aG�@���B-�B0�H@g�?�{@�(�BX��A�G�                                    BxhAi�  T          @��@Z=q@qG�@��\B.(�B?\)@Z=q@z�@���B]G�A�                                    BxhAxz  �          @��H@]p�@���@�p�B$��BE�H@]p�@��@�ffBV\)B
33                                    BxhA�   �          @��@hQ�@n�R@�Q�B)�HB7�@hQ�@33@�{BW=qA�G�                                    BxhA��  T          @���@Vff@z�H@���B+=qBE�H@Vff@{@�Q�B\�\B��                                    BxhA�l  �          @���@\��@vff@�Q�B*��B@�@\��@	��@�\)BZ�A��                                    BxhA�  �          @�G�@Z�H@w�@���B+33BB  @Z�H@
=q@�Q�B[��B ��                                    BxhA��  �          @�Q�@[�@k�@�(�B0��B;��@[�?�Q�@���B_{A�=q                                    BxhA�^            @�  @Y��@`��@�Q�B6��B7@Y��?޸R@��
Bc�A��                                    BxhA�  �          @��@`  @Q�@�p�B<��B,��@`  ?�(�@�{Be��A�                                      BxhA��  �          @�=q@mp�@0  @��HBD�RBG�@mp�?fff@�{Be(�A[�                                    BxhA�P  
�          @�\@`  @G
=@���BB(�B&z�@`  ?�G�@���Bi�A��                                    BxhB
�            @��@U�@Vff@��B@  B4��@U�?�  @�G�BkffA��                                    BxhB�  �          @�\@`��@E@��BBp�B%�\@`��?�p�@���Bi(�A�z�                                    BxhB(B  �          @��@k�@9��@���BA��B{@k�?��@�Bd��A�
                                    BxhB6�  �          @ᙚ@w
=@#33@�=qBC�HB{@w
=?5@��B`A&�\                                    BxhBE�  �          @�  @g
=@&ff@�(�BI��B��@g
=?:�H@�Bh��A6{                                    BxhBT4  
�          @��
@;�@R�\@�  BH{BAG�@;�?�@���Bv�AΏ\                                    BxhBb�  T          @�  @{@u�@�\)B>(�Bd�H@{@ ��@�ffBw\)B�H                                    BxhBq�  �          @�
=@Q�@y��@�p�B<Q�Bj�\@Q�@ff@�p�Bw(�B%�                                    BxhB�&  �          @أ�@
=q@��
@�p�B:p�Byp�@
=q@33@��Bx��B;=q                                    BxhB��  �          @���@�@�=q@�z�B8�HBp@�@  @�ffBu��B0(�                                    BxhB�r  �          @�\)@��@~�R@�\)B?(�Bv��@��@��@�  B}  B3�                                    BxhB�            @أ�@��@���@��B=��Bxff@��@(�@�G�B|=qB6                                    BxhB��  �          @�Q�?��@��@�p�B:�HB�?��@��@�G�B}  BN��                                    BxhB�d  �          @�  ?�33@��\@���B@p�B�(�?�33@p�@\B���BEp�                                    BxhB�
  �          @�Q�?�@�G�@��B"�RB�  ?�@Dz�@��Bh�Blz�                                    BxhB�  �          @���@�@�ff@�z�B,��B�@�@+�@��HBn��BN\)                                    BxhB�V  �          @�Q�@��@�G�@���B3z�B}�R@��@{@�p�Bt=qBDG�                                    BxhC�  �          @�{@�@�=q@�p�B1  B��3@�@!�@��HBrBJ                                    BxhC�  T          @ҏ\@33@w
=@�p�BAp�Bx(�@33@G�@�p�B�33B2                                      BxhC!H  �          @��H@
=@u@�p�BA(�Bt�H@
=?��R@�p�Bp�B-ff                                    BxhC/�  �          @љ�@@o\)@�ffBDz�Bs33@?��@��B��B'��                                    BxhC>�  �          @љ�@	��@`  @��HBLQ�Bi��@	��?�{@�
=B��\BG�                                    BxhCM:  �          @�{@�@dz�@�BG�Bo
=@�?�(�@��HB�G�Bp�                                    BxhC[�  �          @�p�@�@c�
@���BG33Bm=q@�?�(�@��B�ǮBQ�                                    BxhCj�  �          @�p�@p�@b�\@�(�BE�RBg�@p�?ٙ�@���B�u�B(�                                    BxhCy,  �          @Ϯ@�@dz�@�p�BE  Be��@�?��H@��\B�RBz�                                    BxhC��  �          @�{@��@aG�@�BG��Bg�
@��?�z�@��\B�aHB�R                                    BxhC�x  �          @�  @ ��@r�\@��HBA�
Bx{@ ��?�Q�@��HB��HB/p�                                    BxhC�  �          @Ӆ?�(�@�33@�ffBB�B�� ?�(�@(�@���B���B`G�                                    BxhC��  T          @��?\@���@��RBA
=B��q?\@�R@�=qB�
=B^                                    BxhC�j  �          @�{?��H@���@�{B>Q�B�G�?��H@@��HB�B�Bhff                                    BxhC�  
�          @�(�?��@��H@���BE��B���?��@��@ÅB�W
Bj��                                    BxhC߶  T          @��
?���@�  @�=qBH{B�.?���@�@�(�B��BZ33                                    BxhC�\  �          @ҏ\?�\)@|(�@���BF�B�\)?�\)?��R@��B��BK�R                                    BxhC�  �          @�\)?��@z�H@��B?z�B���?��@�\@��B�L�B=�\                                    BxhD�  �          @У�?�\@�=q@���B<\)B�{?�\@(�@�(�B��
BLp�                                    BxhDN  �          @��?�Q�@��@���B:�B�Ǯ?�Q�@G�@��B�ǮBUp�                                    BxhD(�  �          @��H?��@�=q@�z�B3=qB�\?��@p�@��HBz
=BX�                                    BxhD7�  �          @���?�Q�@��@�
=B9Q�B�?�Q�@(�@��\B}\)BB{                                    BxhDF@  �          @ָR@�@n�R@���BC=qBf�@�?�G�@���B��B�                                    BxhDT�  �          @�p�@(�@i��@���BCp�B`�R@(�?�
=@�\)B~\)B
\)                                    BxhDc�  �          @�G�@33@n{@�33B?�HBh��@33?�@��HB}z�B��                                    BxhDr2  �          @��
?�Q�@�  @�p�B4
=B��f?�Q�@
=@��By�BJp�                                    BxhD��  �          @�(�?�{@��@�p�B'��B�� ?�{@.�R@�\)Bp  B^�H                                    BxhD�~  �          @�?��@�@�z�B/B�Ǯ?��@"�\@�z�Bw{BU=q                                    BxhD�$  �          @�=q@�
@Z=q@�\)BR��Bkz�@�
?�\)@��HB�ǮB��                                    BxhD��  �          @�p�?��R@[�@�G�BO\)Bo�R?��R?���@�p�B��fB��                                    BxhD�p  �          @���?�z�@`��@��BM=qBu��?�z�?��@���B��
B\)                                    BxhD�  �          @ʏ\@
=@Z=q@���BJ��Bh�
@
=?�(�@���B�  B�\                                    BxhDؼ  �          @�G�@ff@Q�@�BNBe=q@ff?��@���B�G�B��                                    BxhD�b  �          @�G�?�\)@Y��@�{BOp�Bt�?�\)?�Q�@��\B��)B��                                    BxhD�  
�          @��@ ��@aG�@��\BG��BqG�@ ��?˅@�Q�B�\B�                                    BxhE�  T          @ə�@ ��@S�
@�
=BP=qBj�@ ��?��@�=qB��fB�
                                    BxhET  T          @�=q@�
@J�H@��BT�Bd33@�
?�
=@�33B���A�{                                    BxhE!�  T          @��@�
@G�@��\BV�Bbff@�
?�\)@�33B�aHA�                                      BxhE0�  �          @ȣ�@z�@L(�@��
BL(�BW�H@z�?�  @�{B�W
A�(�                                    BxhE?F  �          @�=q@��@C�
@��BP\)BP  @��?�=q@�  B��A��                                    BxhEM�  "          @���@\)@O\)@�z�BL�\B]ff@\)?��
@�
=B�p�A�z�                                    BxhE\�  �          @ȣ�@
=q@<��@��HBY33BW�@
=q?p��@��B��A�ff                                    BxhEk8  �          @�ff@9��@l��@���B6Q�BO��@9��?޸R@��Bo(�A�                                    BxhEy�  �          @ָR@K�@u�@��B(=qBI33@K�?�(�@�=qB`z�A���                                    BxhE��  �          @�(�@L��@p��@�p�B'z�BFff@L��?�@�\)B_  A���                                    BxhE�*  �          @ָR@HQ�@z=q@��RB&�RBMG�@HQ�@�\@�=qB`�B�                                    BxhE��  �          @�
=@H��@�  @�z�B#G�BO��@H��@	��@�G�B^(�B	��                                    BxhE�v  
�          @ٙ�@U@�Q�@�33BG�BH��@U@
=q@�  BXz�B��                                    BxhE�  �          @�=q@X��@|(�@��B!=qBEG�@X��@�@�G�BYQ�A�(�                                    BxhE��  T          @�
=@Z�H@w
=@��BffBA��@Z�H@�\@��BV�A�z�                                    BxhE�h  �          @ҏ\@E�@{�@���B"{BO�R@E�@
=@��B]p�B	��                                    BxhE�  �          @�G�@;�@�  @�Q�B"�RBW33@;�@
�H@�p�B`Q�Bp�                                    BxhE��  �          @�Q�@2�\@}p�@��HB'ffB[�@2�\@ff@�\)Be�HB�                                    BxhFZ  �          @�(�@&ff@o\)@�ffB1p�B]{@&ff?�{@�Q�BoBp�                                    BxhF   �          @���@Q�@tz�@���B4�Bh�@Q�?��@��
BvG�B�H                                    BxhF)�  �          @�ff@�@g
=@��
BDz�Bk�@�?�=q@��B��RB�                                    BxhF8L  �          @���?���@b�\@�Q�BN{B{�?���?���@�
=B�G�B\)                                    BxhFF�  �          @�?�33@e�@��BPp�B�aH?�33?��H@���B��qB&                                      BxhFU�  �          @��?�=q@fff@��RBK\)B|{?�=q?\@�{B�Q�B�                                    BxhFd>  �          @�(�@%@i��@���B5p�BZff@%?�(�@��Bs�B
=                                    BxhFr�  T          @ʏ\@7
=@g�@��\B,�
BN�\@7
=?�G�@��BgA�ff                                    BxhF��  T          @�Q�@3�
@aG�@��HB/�RBM�R@3�
?�@��HBj(�A��                                    BxhF�0  �          @�  @3�
@W�@��RB5BH�@3�
?�p�@�z�Bn{A�{                                    BxhF��  �          @�Q�@0  @XQ�@��B7ffBK�R@0  ?�(�@�Bp�A�p�                                    BxhF�|  �          @�Q�@(Q�@S�
@�33B=��BN(�@(Q�?�{@�  Bv�HAڣ�                                    BxhF�"  �          @�Q�@-p�@L(�@�z�B?�BFff@-p�?�p�@��Bv�A���                                    BxhF��  T          @���@0  @I��@��B@p�BC�@0  ?�
=@�  Bu�HA�                                      BxhF�n  �          @�=q@(Q�@R�\@�{B@=qBM��@(Q�?�ff@��\By(�A�{                                    BxhF�  �          @ə�@#33@\(�@�z�B==qBU�@#33?���@��HBy\)A���                                    BxhF��  �          @�=q@��@U@���BD{BV@��?�ff@�p�B�A߮                                    BxhG`  �          @���@Q�@Vff@�  BD�RBZ�\@Q�?��@��B�ǮA�\)                                    BxhG  �          @ƸR@�@XQ�@��B?��BY\)@�?�33@�G�B|��A�33                                    BxhG"�  �          @���@"�\@O\)@�=qB@{BO�@"�\?��\@��RBz
=A�z�                                    BxhG1R  �          @�(�@"�\@Mp�@��B@G�BN�
@"�\?�  @�{Bz
=A��
                                    BxhG?�  �          @���@!�@C�
@��BC�
BI@!�?���@�(�B{ffA���                                    BxhGN�  �          @�33@,(�@E@���B?\)BC��@,(�?��@�33Bu�\A�                                    BxhG]D  �          @�(�@%@P��@��B<�BM��@%?��@���Bv�
A�                                    BxhGk�  �          @��@0��@Vff@�33B4Q�BJ33@0��?�Q�@���Bn\)A܏\                                    BxhGz�  �          @�z�@(Q�@XQ�@���B7
=BPQ�@(Q�?���@��Br�HA�                                      BxhG�6  T          @���@Q�@\(�@�Q�B<�B]=q@Q�?���@�\)B|33A�33                                    BxhG��  T          @�z�@Q�@aG�@�p�B8�\B_ff@Q�?�ff@�ByQ�Bp�                                    BxhG��  �          @���@&ff@L(�@���B;�RBKp�@&ff?�G�@�G�Bu�A���                                    BxhG�(  �          @��@�H@@��@��BI33BL��@�H?}p�@��RB��RA�                                    BxhG��  �          @�=q@{@0��@�BY\)BL�@{?&ff@��
B�=qA��
                                    BxhG�t  �          @�=q@5�@>{@��RB=�RB9@5�?��\@�Q�Bq(�A��H                                    BxhG�  �          @��
@>{@@��@�p�B9=qB5��@>{?���@��Bk�A�=q                                    BxhG��  �          @�(�@6ff@L��@�z�B6��BA{@6ff?�  @���Bm��A��
                                    BxhG�f  �          @��
@?\)@7
=@�  B=��B.�@?\)?c�
@�Q�Bm=qA�
=                                    BxhH  �          @��H@K�@AG�@�
=B/z�B.{@K�?�33@��B`��A��R                                    BxhH�  �          @���@:=q@Q�@��B1��BA��@:=q?���@��Bj
=A���                                    BxhH*X  �          @�{@1�@Z=q@��B2�HBK\)@1�?���@��HBnp�A��                                    BxhH8�  �          @ƸR@9��@U�@��B2p�BC�
@9��?�\)@��Bk�A�{                                    BxhHG�  �          @�@:=q@QG�@��B3z�BA33@:=q?��@�G�Bk�\A�G�                                    BxhHVJ  �          @Ǯ@J=q@dz�@���B ��BA�H@J=q?��H@�33B[G�A�\)                                    BxhHd�  �          @�33@N{@p  @\)BBEp�@N{?��@�z�BX=qA��H                                    BxhHs�  �          @�z�@Mp�@j=q@�(�B!G�BC  @Mp�?޸R@�\)B\ffA�(�                                    BxhH�<  �          @�\)@Vff@`��@w�B(�B9{@Vff?ٙ�@�BR��A�p�                                    BxhH��  �          @Ǯ@<��@Z=q@���B.��BD�
@<��?�Q�@���Bh��A���                                    BxhH��  �          @�33@.�R@Z=q@�  B0�BMff@.�R?��H@�  Bn
=A���                                    BxhH�.  T          @��@3�
@`��@z�HB$p�BMff@3�
?�@��Bc�A�z�                                    BxhH��  �          @���@>�R@c�
@u�B{BHp�@>�R?�G�@�B[��A�=q                                    BxhH�z  �          @�Q�@C33@Y��@z=qB"\)B@z�@C33?���@�{B\��A�                                      BxhH�   �          @�Q�@L(�@\��@o\)B  B<��@L(�?�
=@���BTQ�A�ff                                    BxhH��  �          @�G�@A�@U�@���B'Q�B>��@A�?���@���B`�
A���                                    BxhH�l  �          @��@XQ�@J�H@p��B33B,��@XQ�?�z�@��RBP��A��                                    BxhI  T          @�
=@W
=@N�R@l��B��B/\)@W
=?��R@�BOQ�A���                                    BxhI�  �          @�ff@#�
@G
=@�G�B;=qBJ�@#�
?��@�BvQ�A�\)                                    BxhI#^  �          @�{@�
@@  @�ffBQ  B^ff@�
?^�R@�Q�B��HA���                                    BxhI2  �          @�{@�@C33@�(�BM(�B\�@�?n{@�
=B�
=A�Q�                                    BxhI@�  �          @�p�@�@K�@��BE\)B^
=@�?�{@�z�B��A�G�                                    BxhIOP  �          @��?�Q�@HQ�@��
BM�RBh�R?�Q�?�G�@��B��Aۅ                                    BxhI]�  �          @��@�
@Mp�@�{BEp�Be��@�
?�33@��B��\A�                                      BxhIl�  �          @��@�@N�R@��BB��Bd�H@�?�Q�@���B�G�A���                                    BxhI{B  �          @���?���@S�
@�(�BDp�Bs=q?���?�G�@��HB��\B��                                    BxhI��  �          @�G�?ٙ�@U�@�p�BGG�B{��?ٙ�?�  @�z�B�=qB��                                    BxhI��  �          @�  @\)@K�@��B=��B[ff@\)?�Q�@�p�B\)A��                                    BxhI�4  �          @�\)@�@H��@��RB=p�BX{@�?�z�@��
B}�A�                                    BxhI��  �          @�Q�@@J=q@�
=B<(�BU��@?�@�z�B|�A���                                    BxhIĀ  �          @�=q@)��@Mp�@��B1  BI��@)��?��
@���BnG�A�=q                                    BxhI�&  �          @�=q@'�@P  @���B0z�BLQ�@'�?���@���Bn��A�
=                                    BxhI��  �          @�G�@ ��@K�@�z�B6\)BO(�@ ��?�(�@��\Bt�A�G�                                    BxhI�r  �          @���@*�H@>�R@�p�B8�RB@ff@*�H?�G�@���Bqp�A�                                      BxhI�  �          @�\)@*�H@HQ�@\)B0\)BF
=@*�H?�(�@�p�Bl��Aģ�                                    BxhJ�  �          @��@,(�@C33@{�B0  BBp�@,(�?�@��\Bk  A�Q�                                    BxhJd  �          @��@&ff@G
=@|(�B0�RBHz�@&ff?�(�@��
Bm��A���                                    BxhJ+
  �          @�33@"�\@Dz�@|(�B2�RBI�H@"�\?�
=@�33Bp=qAǙ�                                    BxhJ9�  �          @���@�\@R�\@u�B.��B\��@�\?�@�33Bs�HA�z�                                    BxhJHV  �          @��R@��@[�@j�HB)
=Bhff@��?�{@�Q�Br�B
=                                    BxhJV�  �          @�  @z�@l(�@]p�B�HBs  @z�?�Q�@�p�Bk�B,ff                                    BxhJe�  �          @�{?�(�@QG�@xQ�B5�Bk��?�(�?�\)@�(�Bp�BQ�                                    BxhJtH  �          @��@G�@P��@tz�B3�Bh��@G�?���@�=qB|Q�B	\)                                    BxhJ��  �          @�@�@I��@{�B933Bd�R@�?��R@�(�B��A���                                    BxhJ��  �          @���?��@Mp�@z=qB9��Bm�
?��?��@�z�B���B	(�                                    BxhJ�:  �          @�{?�@N{@~{B;��Bq33?�?��
@�ffB�G�B\)                                    BxhJ��  �          @���?�
=@L��@y��B8��Bk?�
=?��@�(�B�.B�R                                    BxhJ��  �          @�(�@�@J=q@w
=B7
=Be=q@�?�G�@�=qB~��A��                                    BxhJ�,  �          @���?�
=@L��@x��B8�RBk�?�
=?��
@��
B�33B(�                                    BxhJ��  �          @�
=@33@A�@��B@�B_�\@33?�ff@��RB�33A�Q�                                    BxhJ�x  �          @��?�p�@C33@�ffBDQ�Bc�?�p�?�G�@��HB�ǮA�                                    BxhJ�  �          @���?�{@N�R@��HB>z�Bp  ?�{?��H@��B��{BG�                                    BxhK�  �          @���@�@P��@{�B5�Be@�?��@�{B~
=B p�                                    BxhKj  �          @��@(�@QG�@u�B0Ba�@(�?�{@�33Bx=qA���                                    BxhK$  �          @��@@QG�@hQ�B(Q�BYz�@?�
=@�p�Bn33A��                                    BxhK2�  T          @�(�?��@O\)@z=qB:p�Bt�?��?��
@��B���B�
                                    BxhKA\  �          @��?��H@P  @y��B;�By  ?��H?��@��B��fBQ�                                    BxhKP  
�          @��
?�\)@O\)@}p�B>G�B}�?�\)?�G�@��RB�\B�H                                    BxhK^�  �          @�?��
@S33@���B?��B�(�?��
?��
@�G�B���Bz�                                    BxhKmN  �          @��?�p�@B�\@���BDG�Bq33?�p�?��@�{B��=A��                                    BxhK{�  �          @�p�?�(�@G�@��\BC�Bt�?�(�?��@���B���BG�                                    BxhK��  �          @�z�?�@>�R@��BF�Bj�H?�?s33@�\)B�aHA�G�                                    BxhK�@  �          @�@G�@<��@��HBC\)B^�\@G�?n{@��RB�A���                                    BxhK��  �          @�33?�
=@G
=@��HBHffB���?�
=?��@���B���B��                                    BxhK��  �          @�=q?z�H@S�
@���BF
=B���?z�H?�G�@��\B���BP��                                    BxhK�2  �          @�=q?
=q@c33@z�HB>=qB��q?
=q?\@��B��{B��                                    BxhK��  T          @��>�ff@\(�@��\BF�B��>�ff?���@��B�{B��
                                    BxhK�~  T          @�33?z�@h��@w
=B9G�B�  ?z�?У�@���B�33B�
=                                    BxhK�$  �          @��=u@fff@x��B<B�  =u?�=q@��B�L�B�u�                                    BxhK��  T          @��\���@aG�@�  BBG�B��þ��?��H@��
B�
=B��                                    BxhLp  �          @��H<��
@a�@�Q�BBB�� <��
?���@�z�B�u�B���                                    BxhL  �          @�z�<��
@Z=q@�p�BJB�Q�<��
?��
@�\)B��=B�(�                                    BxhL+�  �          @�p�>\)@X��@�\)BM33B�G�>\)?�(�@���B��RB�                                      BxhL:b  �          @�z�?\)@?\)@�{B^33B���?\)?G�@�G�B��BX�
                                    BxhLI  T          @�p�?xQ�@<��@�{B\�B���?xQ�?:�H@���B�\Bff                                    BxhLW�  �          @�p�?aG�@K�@���BQ�B�=q?aG�?�G�@�\)B�=qBD�
                                    BxhLfT  �          @��R�B�\@n�R@~�RB;Q�B��B�\?У�@��RB�
=B�#�                                    BxhLt�  �          @�
=��@s33@x��B5p�BÙ���?�p�@���B�Q�B�z�                                    BxhL��  �          @�G���33@y��@z�HB4{B�\)��33?�ff@�\)B�B�B�33                                    BxhL�F  �          @����ff@|��@}p�B3z�B����ff?�=q@�G�B�ǮB��f                                    BxhL��  �          @����\@|��@{�B2z�B��
��\?�@�Q�B�(�B�Q�                                    BxhL��  �          @�z�W
=@���@tz�B*�HB�aH�W
=?�(�@��RB�G�B�\                                    BxhL�8  �          @��\���R@u@���B9(�B�33���R?�Q�@���B�.BȽq                                    BxhL��  �          @�=q���@\)@w
=B0(�B�uþ��?��@�
=B��HBó3                                    BxhLۄ  �          @�녾�{@tz�@�Q�B8�HB�=q��{?�
=@���B�{B��                                    BxhL�*  �          @��ÿ�@���@L(�B�RB�{��@\)@���Bf�C��                                    BxhL��  �          @�\)�\)@��\@)��A��B����\)@   @�BG33C�                                    BxhMv  �          @�z���
@vff@>�RB�B�����
@��@�(�BY(�C	�                                    BxhM  �          @�33�8��@U@B�\B
33C�{�8��?�z�@�ffBNz�C(�                                    BxhM$�  �          @���(�@q�@9��B33B���(�@
=@���BS�HC
=                                    BxhM3h  �          @�  �!�@dz�@7�B(�B��R�!�?�
=@���BR33C�3                                    BxhMB  �          @���@e@333BG�B�B���?�p�@�33BRG�C�H                                    BxhMP�  �          @�����@n�R@:�HB�B�\)���@33@���BVffCp�                                    BxhM_Z  �          @�\)�G�@Z=q@O\)B�B�ff�G�?�{@�p�Bg{C��                                    BxhMn   T          @������H@`  @]p�B#
=B�=���H?���@��Bu�Cٚ                                    BxhM|�  
�          @����   @g
=@k�B&�RB���   ?���@���By�C\)                                    BxhM�L  �          @�����H@s33@g�B!
=B�3���H?��@�Bv�HC	��                                    BxhM��  �          @����   @p��@_\)Bp�B���   ?�@��Br�HC	ٚ                                    BxhM��  �          @�����@x��@S�
B�B�  ���@ ��@�ffBf�C	�                                    BxhM�>  �          @������@z�H@O\)B��B�#����@�
@���Be�
C
=                                    BxhM��  �          @�G���=q@�G�@S�
B{B����=q@Q�@���Bn=qC�H                                    BxhMԊ  �          @�\)��
=@�z�@J=qB��B�(���
=@�\@�Bk{B���                                    BxhM�0  �          @��Ϳ˅@�p�@?\)B�BݸR�˅@��@�G�Bf�HB�W
                                    BxhM��  �          @�\)�޸R@y��@@��B��B���޸R@Q�@�{Bh�
C=q                                    BxhN |  �          @�ff�{@G�@HQ�B{B����{?���@�ffBh\)C@                                     BxhN"  �          @�=q�
=q@4z�@P  B)�\B����
=q?��@�p�Bo��C�R                                    BxhN�  �          @�ff�@B�\@FffB��B�.�?���@�z�Bd�C��                                    BxhN,n  T          @�{�<(�@HQ�@:=qBG�C.�<(�?�p�@�Q�BJQ�C:�                                    BxhN;  T          @�33�6ff@A�@;�Bz�CQ��6ff?���@~�RBM�
C@                                     BxhNI�  �          @�Q��:=q@N�R@=p�B	33C�R�:=q?��@��BM=qC
                                    BxhNX`  �          @�=q�-p�@S33@J�HB=qCaH�-p�?�G�@�=qBYQ�C��                                    BxhNg  �          @�33�0��@Vff@HQ�B
=Cs3�0��?���@��BVp�C^�                                    BxhNu�  �          @��\�ff@\(�@S�
B�
B����ff?Ǯ@�Q�Bg�CaH                                    BxhN�R  �          @��H�0��@^{@=p�B�
C �\�0��?޸R@��RBPC�                                    BxhN��  �          @�G��/\)@Vff@C33B�
C:��/\)?˅@��BU{C�                                    BxhN��  �          @����7
=@`  @0  A��RC33�7
=?�{@�G�BGz�C�H                                    BxhN�D  �          @�z��AG�@q�@
=A�(�C �f�AG�@�@tz�B5�C�f                                    BxhN��  	�          @���9��@|(�@33A��B����9��@p�@uB5Q�C��                                    BxhN͐  T          @�33�HQ�@xQ�?��HA�  C ��HQ�@#33@aG�B$=qC�
                                    BxhN�6  
�          @�=q�Tz�@a�@��A�33C@ �Tz�@Q�@dz�B(�\CJ=                                    BxhN��  �          @����I��@j=q@
=qA��HC���I��@��@fffB+ffCQ�                                    BxhN��  �          @����H��@fff@  A�p�C)�H��@
�H@i��B.Cc�                                    BxhO(  "          @��H�Mp�@h��@\)A�(�CaH�Mp�@��@j=qB,��Cz�                                    BxhO�  
�          @��H�O\)@hQ�@p�A�p�C���O\)@p�@hQ�B+Q�C��                                    BxhO%t  "          @��H�W�@Z�H@A��C���W�?��H@i��B,z�C޸                                    BxhO4            @����Vff@^�R@�A�  C�)�Vff@@a�B'p�C�                                    BxhOB�  
�          @��H�mp�@Y��?�A�{C	��mp�@�@J�HB�C��                                    BxhOQf  "          @��\�g
=@]p�?�A�33C8R�g
=@{@N�RB
=Cc�                                    BxhO`  
Z          @���hQ�@\��?��A�33Cs3�hQ�@�R@K�B�Cff                                    BxhOn�  "          @����n{@S33?�=qA�G�C
k��n{@�@I��B�C�                                    BxhO}X  �          @�33�l(�@^�R?�
=A�(�C���l(�@�
@FffB�C��                                    BxhO��  
�          @���c�
@^�R?У�A�33C�H�c�
@�@C�
B�C�\                                    BxhO��  �          @�  �[�@g
=?�z�A��C�=�[�@�H@H��BG�C��                                    BxhO�J  �          @�
=�h��@Tz�?ٙ�A�Q�C	���h��@	��@C33B�Cu�                                    BxhO��  T          @�Q��n{@S�
?�
=A���C
O\�n{@	��@A�B��C�3                                    BxhOƖ  �          @��R�j=q@S�
?�A��C	޸�j=q@	��@AG�B�RC��                                    BxhO�<  "          @�ff�k�@W
=?��RA�p�C	���k�@�@8��B�
C@                                     BxhO��  �          @���b�\@Mp�?��A�  C	�
�b�\@   @Dz�BQ�C�                                     BxhO�  �          @�
=���R@O\)?��
AX��Ch����R@G�@)��A�C��                                    BxhP.  �          @�33��33@N{?�z�AG�
C�
��33@�
@!�A��C��                                    BxhP�  �          @��\���H@N{?�{A@��C�q���H@�@\)A�
=CQ�                                    BxhPz  
�          @�Q�����@=p�?E�A�RCQ�����@  @�
A��C5�                                    BxhP-   �          @������@8��?!G�@��Cn���@��?�A��\C��                                    BxhP;�  �          @�Q����@.{>��R@W�C����@��?�ffA�33CJ=                                    BxhPJl  
�          @����\)@1G�>�@��C5���\)@{?�p�A��RC�
                                    BxhPY  �          @�G���  @
=q��33�r�\C���  @z�?5@�=qCp�                                    BxhPg�  T          @�  ���R@G������{C:����R@�R?&ff@��
C��                                    BxhPv^  
�          @��
��  @��z����C���  @=q?#�
@ٙ�C�                                    BxhP�  T          @����(�@�Ϳc�
��C{��(�@%�>�p�@���Cٚ                                    BxhP��  �          @�Q�����@\)��  �0  C33����@*=q>�z�@I��C}q                                    BxhP�P  �          @��R���@%�u�)��C�����@/\)>\@�ffC&f                                    BxhP��  �          @�ff���\@,�Ϳp���'33C
=���\@5�>�ff@�ffC޸                                    BxhP��  T          @��R��@$z�fff� Q�C���@,(�>�
=@��C�                                     BxhP�B  
�          @�=q��=q@+��Ǯ���C�
��=q@#33?k�A Q�Cٚ                                    BxhP��  
Z          @�(���z�@+���  �,��C�3��z�@�R?�ffA4z�C�
                                    BxhP�  �          @�33��(�@*�H��  �.{C{��(�@{?��A3�C��                                    BxhP�4  "          @�������@.{�B�\��C������@\)?�\)A@��C�H                                    BxhQ�  �          @�����R@>{>u@#�
C\)���R@   ?�\)A���C�                                     BxhQ�  �          @�����\@E�?!G�@��C�����\@=q@�A��
C�f                                    BxhQ&&  "          @������@J=q>���@QG�C� ����@(Q�?�\A���CW
                                    BxhQ4�  �          @������@E�   ���RC�����@=p�?��
A3�CO\                                    BxhQCr  �          @����  @:=q�B�\�G�C���  @)��?�(�AR�HC�                                    BxhQR  �          @��
���H@1G������L(�Cٚ���H@%�?��A6�\C��                                    BxhQ`�  �          @�33����@'
=�����=qC������@!G�?Tz�A33C��                                    BxhQod  T          @�����@ff��\��Q�C�����@�
?.{@陚C�                                    BxhQ~
  T          @��
��@&ff��
=��Q�C�f��@\)?aG�A=qC�q                                    BxhQ��  T          @��
��{@%���
=���RC5���{@p�?^�RA��CL�                                    BxhQ�V  
�          @����  @(��\��=qC��  @�?W
=AQ�C��                                    BxhQ��  �          @��H��@   �
=q���C�)��@p�?8Q�@���CL�                                    BxhQ��  T          @��\���H@	���(��ҏ\C  ���H@�?   @��
C�R                                    BxhQ�H  T          @��H��z�@ff��=q�:=qC����z�?�p�?E�A��C                                    BxhQ��  �          @�����33@Q�����33CB���33?�Q�?c�
A33C+�                                    BxhQ�  T          @�G���Q�@�
���
�L��C���Q�@�?���A?\)C޸                                    BxhQ�:  �          @�  ��ff@z�<��
>B�\C���ff@G�?���AG�C��                                    BxhR�  T          @�=q���
@(��    <��
CG����
@�
?��\A]Cp�                                    BxhR�  T          @����p�@\)>�  @-p�C���p�@�
?�Ax��C5�                                    BxhR,  �          @�������@��>Ǯ@�  CE����?���?\A�ffC:�                                    BxhR-�  �          @�����ff@���\)���
C  ��ff@
�H?��A4��CE                                    BxhR<x  T          @������
@!G��aG����Cp����
@�
?��
A3�CxR                                    BxhRK  �          @��\��@!녾��R�UC����@
=?s33A$��C=q                                    BxhRY�  �          @��H��(�@)�������C=q��(�@Q�?�AJ�RC�\                                    BxhRhj  "          @�����@.{=u?0��C.���@ff?���Ap(�C                                    BxhRw  �          @�G����@,(�<��
>8Q�Cff���@ff?���Ag�C�q                                    BxhR��  �          @�G�����@!녽u�333Cs3����@\)?�AL��C@                                     BxhR�\  �          @�=q��\)@7�>u@&ffCY���\)@��?�{A�z�C�                                    BxhR�  �          @��H��G�@333>�z�@HQ�CT{��G�@33?�\)A���C
                                    BxhR��  �          @��H����@0  ?�\@�{C�=����@
=q?��
A�{C��                                    BxhR�N  T          @�=q��ff@7�>�(�@�p�C33��ff@33?��
A�=qC��                                    BxhR��  S          @������R@3�
?�\@�=qC�=���R@��?���A��RC��                                    BxhRݚ  
�          @�����?�G����W33C%+����?������C �)                                    BxhR�@  �          @�����{?��\����q��C$�=��{?�  ����ffC�                                     BxhR��  
�          @������?�Q쿏\)�L��C%������?˅�Ǯ��ffC!��                                    BxhS	�  T          @�ff����?�(���{�n�RC������?��?+�@���C��                                    BxhS2  �          @�\)���?��z��ʏ\C�����?�{>Ǯ@��C                                    BxhS&�  �          @�ff���?�ff����eG�C%����?�G������HC �                                    BxhS5~  �          @�{���
?�ff��
=��z�C'�\���
?�\)�:�H�=qC!��                                    BxhSD$  T          @�����=q?������C-8R��=q?\�����C"n                                    BxhSR�  
�          @�=q���R?�����H��C'�\���R?��:�H��ffC!k�                                    BxhSap  T          @����  ?�z�u�%G�C!�H��  ?����
�k�C�                                    BxhSp  "          @�33��ff?�(���ff�b�RC#k���ff?�
=�����p�C                                    BxhS~�  �          @�33��z�?�(���\���HC&���z�?�
=�p���"�HCxR                                    BxhS�b  
�          @�33��G�?����\)�AG�C%{��G�?�(����R�U�C!+�                                    BxhS�  �          @�����=q?�ff���
����C$�f��=q@ �׿fff�G�Cff                                    BxhS��  �          @�
=��33?������R��ffC&���33?��
�333���RC�)                                    BxhS�T  �          @�ff���\?�����h��C#�����\?�׾�G�����C�q                                    BxhS��  �          @��R��=q?�=q���H�V=qC!�\��=q?�(���=q�=p�C��                                    BxhS֠  �          @�\)����?�
=��{�D��C#������?�ff����5C޸                                    BxhS�F  �          @����?�\����C33CxR���@���\)�Q�CE                                    BxhS��  �          @�\)��  ?�׿���H��Cff��  @�ͽu�#�
C+�                                    BxhT�  �          @�p����R?����{�F�HC+����R@
=���
�^�RC�f                                    BxhT8  �          @����?�׿����?�C{��@
�H    =#�
C+�                                    BxhT�  �          @�z���?�\)��G��5��C8R��@�=L��?
=qC��                                    BxhT.�  T          @���Q�?�(���{�F{C (���Q�@�\��G���  CǮ                                    BxhT=*  �          @�{����?޸R����<Q�C�����@�\�L�Ϳ
=C�)                                    BxhTK�  �          @�����?��
�s33�)C�{����@G�=L��?��C�                                    BxhTZv  �          @��R���\?�  �Tz���RC !H���\?�
=>��?�\)C@                                     BxhTi  �          @��
��  @��=#�
>���C#���  ?��?���AF�RCJ=                                    BxhTw�  �          @�p���Q�@!G�?�@�  C���Q�?�33?޸RA�=qC5�                                    BxhT�h  �          @��
��G�@�H>k�@p�C&f��G�?��R?�z�Ar�HCu�                                    BxhT�  �          @�(���=q@
=>���@Mp�C����=q?�33?�Q�Aw�
C�                                    BxhT��  �          @������@
=>L��@�C�\����?���?���Aj=qC�                                    BxhT�Z  �          @��
����@Q콏\)�L��C�f����@ff?�\)AA�Cc�                                    BxhT�   �          @�=q��p�@ ��>�{@n{C� ��p�@   ?�ffA�CǮ                                    BxhTϦ  �          @�  ���
@�R>k�@ ��C����
@�\?�Q�A~ffC:�                                    BxhT�L  �          @�Q�����@��=���?�ffC8R����@z�?��AfffC��                                    BxhT��  T          @�  ��33@!G���\)�O\)CW
��33@{?�Q�ARffCJ=                                    BxhT��  �          @�Q����@!�=u?#�
C8R���@
=q?���Ah  Cٚ                                    BxhU
>  �          @�����p�@�>��@8Q�Cs3��p�?�p�?���A
=C�                                    BxhU�  �          @�����\)@33>�{@n{C���\)?���?���A33C�                                    BxhU'�  �          @�Q���  @��>.{?�C����  ?�\)?��\A`(�C�                                    BxhU60  �          @�ff��  @�#�
���
CB���  ?��?h��A ��CQ�                                    BxhUD�  �          @�{���?��Ǯ���\CB����?�{?�R@�(�C��                                    BxhUS|  �          @�{���
?�(���\��C �����
?޸R>�
=@�33C T{                                    BxhUb"  �          @��R���
?�\���H����C ����
?��
>��@�(�C�3                                    BxhUp�  �          @�\)��z�?�(��#�
���C ����z�?�>��R@W�C��                                    BxhUn  �          @��R����?Ǯ�L���{C"c�����?�  =�Q�?uC ^�                                    BxhU�  �          @�p�����?��H�����J{C&.����?�{��p����
C!��                                    BxhU��  �          @��R��ff?�p���=q�?
=C&
=��ff?�{���R�\��C"                                    BxhU�`  T          @�\)���?�\)����@Q�C'Y����?\�\��z�C#�                                    BxhU�  T          @�
=���?��
�����G33C(T{���?��H��ff���C#�R                                    BxhUȬ  T          @��R��
=?�{��\)�G\)C'aH��
=?��
�������C"��                                    BxhU�R  �          @�ff��ff?�{����K�C'aH��ff?��
��(���  C"�                                    BxhU��  �          @�  ��Q�?�(��z�H�,��C&B���Q�?�ff�k��"�\C"�                                    BxhU��  �          @�ff��\)?��
��=q�?33C(E��\)?�Q�����  C#��                                    BxhVD  �          @�ff���R?�G��L����C%�����R?��R�u��RC#O\                                    BxhV�  �          @�����
?�������Q�C"u����
?У�>�  @5C!s3                                    BxhV �  �          @�(����?�ff�xQ��.=qC%����?�{�8Q��   C!��                                    BxhV/6  �          @����33?�녿z�H�/
=C#�q��33?�Q�\)��ffC �                                     BxhV=�  �          @�p����?�
=�L����HC#�{���?��=#�
>��C!^�                                    BxhVL�  �          @���(�?��H�����\)C �3��(�?�
=?�\@�{C!�                                    BxhV[(  �          @�����?��
�0����C"������?�>8Q�?���C!5�                                    BxhVi�  �          @�ff���?�  �:�H���C%�����?���    �#�
C#ٚ                                    BxhVxt  �          @�{���?��\������C%�����?�33=�?���C$W
                                    BxhV�  �          @�p���ff?�\)��R��{C$}q��ff?�  >#�
?޸RC#!H                                    BxhV��  �          @�ff��
=?�녿#�
��G�C$c���
=?\>��?ٙ�C"�q                                    BxhV�f  �          @���{?�{�5��{C$�
��{?��=�\)?Q�C"��                                    BxhV�  �          @�����R?�p��:�H�G�C&)���R?�    ��C#�R                                    BxhV��  �          @�p���?�(��s33�)�C&)��?��
�L���\)C"�R                                    BxhV�X  �          @���Q�?�=q�5���HC'޸��Q�?���u�.{C%�)                                    BxhV��  �          @�(����?�  �&ff��G�C(�����?�Q�L�Ϳ
=C&�\                                    BxhV��  �          @�z���  ?xQ�(����(�C)
=��  ?����
�\(�C&�
                                    BxhV�J  �          @����?aG��#�
��33C*)���?�=q���Ϳ���C'�f                                    BxhW
�  �          @���G�?xQ�(�����C))��G�?���\)�Q�C&�                                    BxhW�  �          @�p����?�p�������C&#����?��>��?�33C$�3                                    BxhW(<  T          @����?�G��
=q����C%޸���?�{>.{?��C$Ǯ                                    BxhW6�  �          @�����?�{�����{C'}q����?��\=L��?�C%ٚ                                    BxhWE�  �          @�p�����?��ÿ(�����C(�����?�p�<��
>L��C&B�                                    BxhWT.  �          @�z���G�?aG��
=q��=qC*
=��G�?�����
�B�\C(Y�                                    BxhWb�  �          @�(�����?W
=�
=��(�C*�����?��\���
�k�C(}q                                    BxhWqz  �          @�����G�?^�R�z���\)C*0���G�?�ff�u��RC(G�                                    BxhW�   �          @�z�����?^�R������C*33����?���L�Ϳ��C(T{                                    BxhW��  �          @�(����?�  �(���(�C(�����?����
�W
=C&Ǯ                                    BxhW�l  �          @��H��?��\�.{��p�C(J=��?�p��u�333C&�                                    BxhW�  �          @����z�?�녿#�
��G�C&޸��z�?��<�>�{C%�                                    BxhW��  �          @�=q���
?����333��p�C&����
?��<#�
>#�
C$\                                    BxhW�^  �          @�=q���H?���@  ��
C$�=���H?�G�<�>��
C"��                                    BxhW�  �          @�33���?�p��+���  C%�����?��=L��?��C$#�                                    BxhW�  
�          @��\��z�?�Q�+����HC&G���z�?�\)<�>\C$\)                                    BxhW�P  �          @��\��z�?��H�&ff��C&���z�?�\)=u?+�C$Q�                                    BxhX�  �          @��H��z�?��
�����Q�C%Y���z�?�z�>\)?ǮC#�R                                    BxhX�  �          @��\��  ?�G��xQ��/\)C"T{��  ?���u�!G�CO\                                    BxhX!B  �          @��H��Q�?���p���*�\C"{��Q�?�ff���
�uC=q                                    BxhX/�  �          @��\����?�  �Y�����C"������?�p�=#�
>�C !H                                    BxhX>�  �          @��H����?�=q�O\)��HC!�3����?�\=�G�?��RC��                                    BxhXM4  �          @�����\)?\�Y���
=C"(���\)?޸R=L��?�CǮ                                    BxhX[�  �          @�����\)?��
�W
=��C"���\)?޸R=u?5C�=                                    BxhXj�  �          @�G�����?�z�L���=qC#������?�\)<�>\C!E                                    BxhXy&  �          @�=q���
?�Q�=p���C&:����
?�33���
��  C#�                                    BxhX��  �          @��\��(�?�z�O\)��C&�f��(�?�z��G���Q�C#�)                                    BxhX�r  �          @������?��R�B�\�	G�C%ٚ����?������
�uC#�                                     BxhX�  �          @��
���?��H�=p��C&#����?����
�8Q�C#�)                                    BxhX��  �          @��
��{?�33�:�H��C&���{?�{�#�
��ffC$��                                    BxhX�d  �          @��
��{?�=q�G����C'�3��{?�=q������C$�R                                    BxhX�
  �          @����?�{�B�\�z�C'Y���?�����
�n{C$�\                                    BxhX߰  �          @����p�?�z�@  ��\C&�R��p�?��׽L�Ϳ�\C$T{                                    BxhX�V  �          @�33���?�\)�L���G�C'����?��׽�G����
C$T{                                    BxhX��  �          @�33��p�?��ÿG����C'� ��p�?��ý���\)C%�                                    BxhY�  �          @����{?��Ϳ333���\C'z���{?�ff���\C%E                                    BxhYH  �          @�=q��(�?��������C$���(�?��>��
@k�C$�                                    BxhY(�  �          @�=q���
?�Q�B�\�	G�C&G����
?�z�#�
��(�C#޸                                    BxhY7�  
\          @�=q���?��\�8Q��\)C(E���?�  ���Ϳ���C%�                                     BxhYF:  �          @�=q��p�?c�
�O\)�33C)����p�?�
=�u�*�HC&�                                     BxhYT�  �          @��\��{?xQ�5� (�C(�H��{?�����G���p�C&aH                                    BxhYc�  "          @�=q��p�?p�׿8Q��33C)8R��p�?�
=�\)�ǮC&��                                    BxhYr,  
(          @�G����H?����5��C&����H?�33<#�
>\)C#��                                    BxhY��  
Z          @�������?�(�������
C"�H����?�  >�33@�G�C"��                                    BxhY�x  
�          @�{��?�=q�W
=���C!W
��?��H?#�
@�z�C"��                                    BxhY�  �          @���{?��R�����u�C"\)��{?���>�@�z�C"�{                                    BxhY��  �          @�����H?�녾���  C#�q���H?�Q�>���@X��C#n                                    BxhY�j  
*          @�=q���?��ÿ
=��(�C$�{���?�
=>8Q�?��RC#��                                    BxhY�  T          @��H��z�?�p��&ff��C%����z�?��=�\)?Q�C$(�                                    BxhYض  �          @����p�?��ÿW
=��HC'�3��p�?��;.{����C$��                                    BxhY�\  "          @�����
=?���J=q��C(���
=?������p�C%:�                                    BxhY�  "          @������?��
�E��	p�C(Q����?��
����
=C%��                                    BxhZ�  
�          @�(����?Y���W
=�ffC*L����?�zᾏ\)�G�C&��                                    BxhZN  
�          @�z���Q�?Tz�Q��
=C*����Q�?��׾�\)�FffC'B�                                    BxhZ!�  
�          @�(����?Tz�O\)�{C*�����?��׾�=q�AG�C'@                                     BxhZ0�  T          @�33��
=?W
=�Q��
=C*aH��
=?�녾�=q�>�RC'\                                    BxhZ?@  �          @��
��\)?^�R�L����C*#���\)?�33�u�+�C&��                                    BxhZM�  T          @����\)?\(��@  ��RC*+���\)?�\)�L���p�C'E                                    BxhZ\�  "          @��\��ff?Y���=p���RC*5���ff?�\)�L���{C'L�                                    BxhZk2  �          @�33��\)?O\)�@  ��HC*��\)?�=q�aG�� ��C'�                                    BxhZy�  �          @�33��\)?L�Ϳ@  �  C*�f��\)?��þu�(��C'ٚ                                    BxhZ�~  �          @��\��
=?J=q�:�H�  C*�R��
=?�ff�aG��p�C(
=                                    BxhZ�$  �          @��\��\)?O\)�#�
��Q�C*����\)?��\�\)���
C(ff                                    BxhZ��  T          @�����?J=q�5��\)C+  ���?���L���\)C(5�                                    BxhZ�p  "          @��H��
=?Y���=p���\C*T{��
=?�{�L����C'k�                                    BxhZ�  �          @�����{?8Q�E��Q�C+���{?��\��z��QG�C(^�                                    BxhZѼ  "          @�G����?\(��:�H�p�C*\���?�{�8Q���C'8R                                    BxhZ�b  
�          @��R���\?aG��0����
=C)�����\?�{���ǮC'�                                    BxhZ�  
�          @�ff���\?Y���!G���G�C*����\?�ff���Ϳ�z�C'��                                    BxhZ��  �          @���=q?J=q�+���Q�C*����=q?��\�#�
��C({                                    Bxh[T  "          @�{��=q?J=q�333�33C*�3��=q?���L����C'�{                                    Bxh[�  
�          @���=q?@  �333�C+#���=q?�  �W
=��C(B�                                    Bxh[)�  T          @�  ���?+��(����C,8R���?k��aG��"�\C)n                                    Bxh[8F  T          @�
=��(�?5��R����C+�q��(�?n{�.{��z�C)B�                                    Bxh[F�  T          @��R���?aG���\��C)�3���?��\<��
>8Q�C(!H                                    Bxh[U�  �          @�ff��=q?�G��   ����C((���=q?�\)=�Q�?��C&��                                    Bxh[d8  �          @��R����?�{�   ����C&�q����?��H>\)?�33C%޸                                    Bxh[r�  �          @�{���?���
=q��
=C$���?���>W
=@��C#�q                                    Bxh[��  
�          @����?��׿�\��  C#�)��?�Q�>�=q@J=qC"ٚ                                    Bxh[�*  �          @�z�����?��
=q���C#�����?��R>�=q@I��C":�                                    Bxh[��  
�          @������?�(�����33C"u�����?��
>��R@hQ�C!��                                    Bxh[�v  #          @��R��ff?��&ff��\C#.��ff?Ǯ>.{?��RC!�3                                    Bxh[�  �          @�p����R?���&ff����C$�����R?�Q�=�G�?��
C#
=                                    Bxh[��  T          @����  ?���0����C'p���  ?�G��L�Ϳ
=qC%�                                    Bxh[�h  "          @�����  ?Q녿Tz��z�C*33��  ?��׾�z��X��C&��                                    Bxh[�  "          @�������?.{�L���
=C+�)����?�  ��{�|(�C(0�                                    Bxh[��  
Z          @��
���?E��L���\)C*�����?��þ�z��Z�HC'G�                                    Bxh\Z  "          @�33���?�R�W
=��RC,�{���?u�������RC(��                                    Bxh\   
�          @��\��
=?5�B�\�G�C+}q��
=?�  ��z��\(�C(\                                    Bxh\"�  
)          @������\?����&ff����C%W
���\?�{=u?B�\C#xR                                    Bxh\1L  
�          @��H��p�?���+���
=C&�f��p�?��
���
�8Q�C$�3                                    Bxh\?�  	�          @�33���R?�  ����  C(���R?���=u?=p�C&��                                    Bxh\N�  
�          @�z����?}p��:�H�	C(&f���?�p�����{C%s3                                    Bxh\]>  T          @������?��
�z����HC$�H����?��>#�
?�C#Q�                                    Bxh\k�  �          @�����  ?��
��
=��G�C!8R��  ?��
>�(�@���C!@                                     Bxh\z�  
�          @�33��?��Ϳ+���C&�=��?���#�
����C$�H                                    Bxh\�0  
�          @����Q�?�G��(����C(���Q�?����#�
��C%ٚ                                    Bxh\��  �          @��
��ff?������Ϳ��C$���ff?�=q?�@߮C%�                                    Bxh\�|  T          @�z���?��ü��
����C"n��?�33?(��A
�HC$�f                                    Bxh\�"  
�          @����R?�33��\)�c�
C!�����R?�p�?(��A��C#�f                                    Bxh\��  �          @���w�?���=#�
?�C ���w�?��?:�HA$z�C#�{                                    Bxh\�n  
Z          @�{�w�?�=q>�p�@��
C�w�?�z�?���A{
=C#B�                                    Bxh\�  T          @�(��tz�?��R>��@��HC�3�tz�?��
?�A�  C$�)                                    Bxh\�  �          @�
=�w�?���?\)@�ffC�3�w�?�ff?��A��RC$                                    Bxh\�`  
�          @����{�?�ff?!G�A\)CxR�{�?�  ?�=qA��
C%��                                    Bxh]  	�          @�z��|(�@�=���?�G�CE�|(�@�\?��A�z�C��                                    Bxh]�  
�          @���p  @7
=>�{@�=qC��p  @�?�G�A��\C��                                    Bxh]*R  
�          @�����=q@�?fffA/
=CǮ��=q?��R?�Q�A�33C�f                                    Bxh]8�  �          @����tz�@�?�33Ae�C#��tz�?�Q�@p�A�33CJ=                                    Bxh]G�  �          @���l��@�?�
=A��C���l��?��\@�A��HC!�                                    Bxh]VD  T          @��H�s33@��?��HAu�CJ=�s33?��@��A�\)C �                                    Bxh]d�  
�          @�p��q�@ff?���A�\)C)�q�?��@��A��RC �)                                    Bxh]s�  T          @��\�w�@�?�{Ac33C��w�?��H@G�A��
C"�)                                    Bxh]�6  �          @�
=��33?}p��W
=�&ffC'ٚ��33?u>��
@|��C(8R                                    Bxh]��  
�          @�{��33?#�
��R��  C,s3��33?^�R�L����C)��                                    Bxh]��  
�          @�  ��?!G����ҏ\C,����?Tz�.{��
=C*aH                                    Bxh]�(  	�          @�Q���{?z����G�C-O\��{?E������\C++�                                    Bxh]��  �          @�ff���
?333����
=C+�H���
?\(���Q�}p�C)��                                    Bxh]�t  "          @�����H?�R����z�C,� ���H?L�;�����C*��                                    Bxh]�  
Z          @�p����H>�׿8Q��\)C.u����H?E��\��{C*�                                    