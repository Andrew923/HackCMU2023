CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230123000000_e20230123235959_p20230124021100_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-24T02:11:00.125Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-23T00:00:00.000Z   time_coverage_end         2023-01-23T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        8   records_fill         h   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx]��   �          @��R�HQ�@5@e�B!  C	���HQ�?�p�@���BR��C��                                    Bx]�٦  �          @��R�U@5�@N�RB�C��U?�{@�=qBB  C��                                    Bx]��L  �          @�\)�U�@ff@��HB8\)C�U�>�33@��\BWp�C.�                                    Bx]���  �          @�=q�/\)@�
@�BW{C&f�/\)=��
@��Bw
=C2c�                                    Bx]��  T          @��R�XQ�@Vff@b�\BffCQ��XQ�?��H@��BH��C.                                    Bx]�>  �          @��ÿL��@>{@�(�BlffB�\�L��?=p�@�(�B�
=C	\                                    Bx]�"�  T          @��ÿ�@��@|(�B%33B߮��@�
@��Bv
=B��q                                    Bx]�1�  �          @�  ��@���@VffB	�B�.��@(��@�  BS�RC��                                    Bx]�@0  T          @�����R@~{@���BAffB��Ϳ�R?�z�@��RB�=qB��                                    Bx]�N�  �          @��H�:�H@�  @�Q�B?�RBȸR�:�H?�Q�@��RB�
=B�L�                                    Bx]�]|  �          @���<#�
@��@W�B��B���<#�
@Y��@��\B`��B��                                     Bx]�l"  �          @�\)=��
@��
@B�\A��
B�G�=��
@q�@�(�BQ
=B��=                                    Bx]�z�  �          @����\@�33@w�B�B�\)��\@ ��@�Q�Bn��B�p�                                    Bx]��n  �          @��H=�@��H@z�A�(�B�� =�@�  @���B5�B��3                                    Bx]��  "          @���\)@�{@Y��B	z�B��\)@1�@��BV�RC ޸                                    Bx]���  �          @�33��\@��H@p�A�=qB�aH��\@o\)@��B6��B��                                    Bx]��`  �          @�ff��@��@��Be=qC��>�@�G�B�ffC0��                                    Bx]��  �          @�\)��z�@p�@�z�Bo��B��{��z�>B�\@�p�B�{C.Q�                                    Bx]�Ҭ  �          @�\)�{@��@��\Bo��C��{���
@�  B���C6)                                    Bx]��R  �          @�Q���@c�
@���B3�RB�����?�\)@��HBu=qC=q                                    Bx]���  �          @�Q���@�z�@R�\A�33B�����@O\)@�p�BO  B�u�                                    Bx]���  �          @ə��*=q@�=q@AG�A��
B����*=q@Q�@�z�B>C
=                                    Bx]�D  �          @����$z�@��H@!G�A�
=B� �$z�@n�R@�G�B-��B�#�                                    Bx]��  �          @ȣ׿�
=@���?�{AqBٸR��
=@�33@j�HB\)B�{                                    Bx]�*�  �          @�Q�O\)@���?�{Ar{Bè��O\)@�=q@p��B�B�{                                    Bx]�96  �          @�z῁G�@�
=?W
=@�{B�G���G�@�@E�A�B�#�                                    Bx]�G�  �          @�(�    @��@QG�Bp�B�      @\)@��\Bu��B�                                      Bx]�V�  �          @�=q����@�33@xQ�B \)B�aH����@0  @�33Bz�B�L�                                    Bx]�e(  �          @�녿�=q@�z�@Tz�B�B�녿�=q@O\)@�{B]z�B��f                                    Bx]�s�  �          @�Q��,(�@l(�@dz�B��B�33�,(�@�\@��RBY�C޸                                    Bx]��t  �          @�=q�p  @E@fffB  C���p  ?�(�@�Q�B@C�{                                    Bx]��  �          @�����{@,(�@W
=B{CE��{?���@�z�B.{C$\                                    Bx]���  "          @�����z�@
�H@C�
A��C���z�?Tz�@j=qBp�C)�)                                    Bx]��f  "          @������@/\)@�RA�
=C����?�=q@UB(�C!�                                    Bx]��  �          @����33@1G�@1G�A�ffC�q��33?��R@g�B(�C"                                    Bx]�˲  �          @�����  @L��@?\)A�ffC���  ?�@\)B#{C��                                    Bx]��X  �          @�(���@G�@J=qA�  C=q��?�@��
B)��C.                                    Bx]���  �          @�z���33@\(�@'�A�{C�f��33@�@n�RB
=Cc�                                    Bx]���  �          @�Q���  @k�@33A�
=C��  @!G�@a�B	�RC�=                                    Bx]�J  �          @Ǯ���\@[�@=qA��RC!H���\@  @b�\B
��C��                                    Bx]��  T          @�=q��
=@>�R?�Q�A�  C���
=@
=@,��A�z�C��                                    Bx]�#�  �          @��H����@Dz�?��Aj�RC������@  @%A�  C޸                                    Bx]�2<  T          @�Q���z�@W
=?�Q�A�
=C���z�@�@E�A�(�C0�                                    Bx]�@�  �          @�33�^�R@`  @~{Bp�C�
�^�R?�(�@�  BP�C�                                    Bx]�O�  �          @�{�q�@[�@uB�HC	� �q�?��H@��BF{C�H                                    Bx]�^.  "          @��E@R�\@�=qB5�C.�E?�ff@�\)Bj33C)                                    Bx]�l�  :          @�z��0��@R�\@�\)B?�C��0��?�  @�(�Bw  C��                                    Bx]�{z  n          @�(��#�
@P  @�33BF
=C @ �#�
?�@�
=B(�C�=                                    Bx]��   T          @˅�,��@>�R@�BK33C@ �,��?aG�@�{B}��C!�                                    Bx]���  �          @�33�33@7�@���BY�C ���33?333@��B�W
C#�                                    Bx]��l  "          @�=q�\)@1G�@��BW��C�R�\)?!G�@���B�\C%��                                    Bx]��  T          @�=q�>{@AG�@��HB=z�C��>{?�ff@�z�Bo  C �                                     Bx]�ĸ  "          @���{@C�
@�  BS��B���{?p��@���B�ǮC\                                    Bx]��^  l          @��H�8Q�@5@��BI
=C\)�8Q�?J=q@��\Bwp�C$��                                    Bx]��  
�          @���<��@Fff@��B<p�C���<��?���@�{BoffC�                                    Bx]��  
�          @ə��<��@H��@�=qB:��C=q�<��?�
=@�p�BnG�C.                                    Bx]��P  
Z          @�
=�:�H@Dz�@��
Br��BΣ׿:�H?8Q�@��
B��RC8R                                    Bx]��  �          @�Q�L��@^�R@�Bb��B�\�L��?�@��HB�B���                                    Bx]��  
�          @�\)�8Q�@`  @��
BaffB�B��8Q�?�(�@ə�B���B�#�                                    Bx]�+B  
�          @˅�P  @\)@��HBF�C���P  >��H@�p�Bj��C+�                                     Bx]�9�  �          @��
�q�@{@�\)B3(�C޸�q�?��@��\BT
=C*��                                    Bx]�H�  "          @�{�=q@.{@�\)B\�
C�
�=q?��@��
B�G�C')                                    Bx]�W4  
�          @�ff��@ff@�z�BtC�=��=u@ÅB�L�C2\)                                    Bx]�e�  �          @������@��@��\BsG�C
=��ͽL��@��B���C5c�                                    Bx]�t�  �          @�
=��\@\)@�Q�Bi��C���\>�  @�G�B�Q�C-�)                                    Bx]&  �          @��)��@7�@�33BR33C���)��?@  @��B�G�C$)                                    Bx]�  "          @�
=�ff@ff@���BlG�C��ff=���@���B�B�C1�
                                    Bx] r  
�          @�\)�޸R?���@�z�B�.C�ÿ޸R����@�{B��fC@��                                    Bx]¯  �          @�\)���?�z�@��B�z�C G�����8Q�@�  B�CP�)                                    Bx]½�  �          @�G��ٙ�@�R@��B�B�B��{�ٙ��\)@��B���C8��                                    Bx]��d  �          @�G��G�@�@��B�
C�=�G�����@�{B��\C<}q                                    Bx]��
  "          @У׿���@z�@���B��HCQ���þ�\)@ǮB�#�C<��                                    Bx]��  �          @������@
�H@�  BxQ�C����þ�@���B�� C7s3                                    Bx]��V  
�          @У׿��@��@��HB~�C W
�����\)@�Q�B��C6aH                                    Bx]��  �          @��ÿ�@�@���Bz(�B��3��=L��@�  B��C2Y�                                    Bx]��  "          @�  �˅@��@�33B�W
B�G��˅��\)@ȣ�B�u�C6��                                    Bx]�$H  �          @�Q��33@8��@��\Bl��B�=��33?�R@�  B�u�C��                                    Bx]�2�  �          @У׿�
=@1G�@�Q�By��B�  ��
=>�G�@�(�B��RC�=                                    Bx]�A�  T          @�Q쿴z�@3�
@�ffBtQ�B�k���z�?   @ʏ\B�W
C u�                                    Bx]�P:  
Z          @�Q쿾�R@'�@�Q�ByffB�\)���R>���@�=qB��HC(xR                                    Bx]�^�  �          @�\)���@1�@���Br��B�녿��?   @���B��C"(�                                    Bx]�m�  �          @�G����@C33@��
Bmp�B�{���?B�\@�33B��RC��                                    Bx]�|,  �          @��H��  @Dz�@��
Bj��B�#׿�  ?G�@˅B�.C��                                    Bx]Ê�  T          @��
��Q�@Tz�@�G�Bcp�B��f��Q�?��@��
B��
C\                                    Bx]Ùx  "          @Ӆ�޸R@_\)@��HBW\)B�  �޸R?��
@�  B��
C�f                                    Bx]è  	�          @ҏ\�s33@�ff@�
=BDffB͊=�s33@�@��B�W
B�=q                                    Bx]ö�  :          @љ��333@�=q@���BJ�HB�k��333?�
=@�{B�L�Bۨ�                                    Bx]��j  
�          @љ����
@Z=q@�\)Bc�B��Ϳ��
?�z�@��HB�z�C�                                    Bx]��  �          @Ϯ��Q�@S�
@�z�BT��B���Q�?���@��B�W
CW
                                    Bx]��  T          @�����\@Dz�@���BX�B�ff��\?h��@���B��qC5�                                    Bx]��\  �          @У��G�@7�@��Be�B���G�?(��@��B�  C!�q                                    Bx]�   
�          @�녿���@*�H@��Bz�\B�R����>�Q�@�(�B�ǮC%Q�                                    Bx]��  �          @�=q��33@8��@��RBr�B��)��33?��@��
B�
=C�                                    Bx]�N  T          @�G����@'
=@��HB�B�B�LͿ��>���@�z�B��fC$k�                                    Bx]�+�  
�          @У׿�(�@��@�{Bu��C����(�>��@�p�B�� C/�
                                    Bx]�:�  :          @Ϯ�&ff@p�@���Ba�RC�=�&ff>���@�p�B�B�C,�q                                    Bx]�I@  n          @Ϯ�(�@@��@���B[��B���(�?aG�@�G�B�\)C)                                    Bx]�W�  T          @�G��	��@U�@�ffBRB����	��?��H@���B���C}q                                    Bx]�f�  
�          @�녿��
@aG�@�
=BS��B陚���
?��@�z�B���C�q                                    Bx]�u2  �          @�=q���R@n�R@�Q�BU  Bب����R?���@�Q�B��{C 33                                    Bx]ă�  
�          @�33��  @p��@��
BL  B�Ǯ��  ?�33@�(�B�33C�=                                    Bx]Ē~  �          @ҏ\��=q@O\)@�  Bc33B���=q?��
@ə�B��3Cٚ                                    Bx]ġ$  T          @��ÿ�z�@P��@�\)BdQ�B�3��z�?��@�G�B��3C�q                                    Bx]į�  
�          @�  ����@I��@��RBd�\B�\����?xQ�@�\)B��HC@                                     Bx]ľp  �          @�\)��@3�
@�
=Bw��B�W
��?\)@��HB��)C��                                    Bx]��  ;          @�Q����@.{@���BoB��\����?   @�  B�ffC$�                                     Bx]�ۼ  
�          @�
=�333@=q@�ffB�8RB�G��333=�G�@�p�B�k�C*��                                    Bx]��b  T          @�Q�xQ�@'
=@�33B�
=Bܔ{�xQ�>�{@�z�B���C ��                                    Bx]��  
�          @Ϯ��(�@#�
@��\B�{B����(�>���@˅B��{C&&f                                    Bx]��  
�          @�{����?ٙ�@\B�.B�G������z�@���B�
=CN                                    Bx]�T  T          @���ff?�33@��B��qC \)��ff�!G�@ǮB��CM�f                                    Bx]�$�  
�          @�ff�Q�?L��@�p�B�
=C!� �Q쿯\)@��B�k�CQ�f                                    Bx]�3�  �          @�z��{?�\)@��
B�u�C
�q��{�z�@��B�\)CEaH                                    Bx]�BF  
�          @�z��ff>�@ÅB��C%����ff�޸R@��
B��RC`\                                    Bx]�P�  "          @����
?0��@�(�B�aHC����
���@�\)B��C\�\                                    Bx]�_�  
�          @�p�����?=p�@\B�ǮC����Ϳ�(�@�ffB�CZ}q                                    Bx]�n8  
�          @�33��\)�z�@��B!�C;���\)���R@eB�HCJ                                    Bx]�|�  "          @�G����H>��R@�z�B'�HC0&f���H��
=@~�RB {CBaH                                    Bx]ŋ�  
Z          @������>k�@�z�B@\)C0ٚ�����
=@�p�B4��CF�3                                    Bx]Ś*  �          @���#�
?�{@��
Bn  C\�#�
�#�
@�B�aHC7�
                                    Bx]Ũ�  T          @ʏ\��z�?Tz�@�Q�B��{Cs3��zΎ�@�p�B�B�CZ�H                                    Bx]ŷv  �          @�=q��\>��R@�z�B�ffC+E��\���
@�(�B��\C](�                                    Bx]��  �          @�{�p�>�\)@�Q�B�L�C,�3�p�����@�\)B}{C[�                                    Bx]���  
Z          @�����
?}p�@��B��)C녿��
����@���B��qCV\                                    Bx]��h  
�          @�p��ff?��H@�B��{C  �ff�s33@�\)B��fCLL�                                    Bx]��  
Z          @ʏ\�,��?p��@��
B|Q�C �
�,�Ϳ��@��HBz{CJ�                                    Bx]� �  
�          @��Ϳ���?��\@�{B��C�������c�
@�Q�B��3CL^�                                    Bx]�Z  "          @�{��  ?�@�Q�B��fC���  �E�@�(�B�Q�CK�                                    Bx]�   "          @���p�?��
@��B�W
C=q��p��Ǯ@��B�ǮC@��                                    Bx]�,�  
Z          @�\)��@	��@��\B�k�C{���#�
@ƸRB�u�C4^�                                    Bx]�;L  T          @�  ��(�@{@�=qB~�B�p���(�>�z�@�=qB�W
C(�{                                    Bx]�I�  �          @У׿k�@(��@�(�B�33Bڔ{�k�>�(�@�B�
=C!H                                    Bx]�X�  �          @Ϯ�c�
@.{@���B(�B�#׿c�
?�@�(�B��{C�                                    Bx]�g>  
�          @�
=�xQ�@\)@��
B�  B�LͿxQ�>��R@��
B��C"^�                                    Bx]�u�  
�          @θR���
@ff@�
=B�.B�\)���
��Q�@�=qB�aHC9=q                                    Bx]Ƅ�  �          @�z��@Fff@�  Bq{B�녿�?�G�@�\)B�k�B�L�                                    Bx]Ɠ0  �          @����@j�H@���B\ffB�����?У�@ƸRB�z�B���                                    Bx]ơ�  �          @���
=@Y��@�z�Be\)BǏ\�
=?��@�\)B��{B�L�                                    Bx]ư|  "          @�(��!G�@Z=q@��\BcB�  �!G�?�\)@�B��=B�=                                    Bx]ƿ"  "          @�33�h��@c�
@�p�BZz�BЙ��h��?���@�=qB��
B��                                    Bx]���  T          @�(���@_\)@���Ba�B�#׾�?��H@�B�#�B�B�                                    Bx]��n  �          @��
�+�@l(�@�z�BWQ�BȮ�+�?��H@��HB�aHB���                                    Bx]��  T          @��Ϳ���@a�@��RBZ�\Bօ����?��@ÅB���B�                                    Bx]���  
�          @������@\(�@���B]p�B�ff����?�
=@�(�B�{C�)                                    Bx]�`  �          @��Ϳ.{@n�R@�z�BV�\Bș��.{?�  @�33B��
B�L�                                    Bx]�  T          @�{�   @\)@�Q�BM  B�\)�   @33@�=qB��Bϙ�                                    Bx]�%�  T          @�{��=q@a�@�{BX
=B�(���=q?Ǯ@��HB��
Ch�                                    Bx]�4R  
�          @�  ���R@s�
@��BN��B�33���R?�{@���B�aHB�p�                                    Bx]�B�  �          @�녿s33@���@�
=BE��Bͨ��s33@{@\B���B�{                                    Bx]�Q�  
�          @�=q�L��@z�@��\BJ��C��L��?��@��\Bj��C*5�                                    Bx]�`D  "          @θR�=p�@3�
@�
=BJffCz��=p�?xQ�@�(�Bt�C!޸                                    Bx]�n�  "          @Ϯ�>{@)��@��
BP�C
:��>{?E�@��RBw
=C%k�                                    Bx]�}�  T          @�z��c33@�@�ffBB�RC0��c33>��@�z�B\C-^�                                    Bx]ǌ6  "          @��H����?G�@�ffB&�C*�=���Ϳ.{@�
=B'�C<W
                                    Bx]ǚ�  �          @˅���?��\@�G�B6�C#W
�������@�ffB?�C8h�                                    Bx]ǩ�  
�          @�=q���\?�33@��
B/\)C"!H���\���@��\B:z�C5�                                    Bx]Ǹ(  "          @�33��
=?�z�@��\B,��C%aH��
=��33@�
=B3�RC8u�                                    Bx]���  
(          @�(�����?s33@�Q�B4{C'Ǯ���Ϳ��@�=qB7(�C;�\                                    Bx]��t  T          @ə����?��@��HB.C&&f�����
=@��RB4��C9n                                    Bx]��  �          @�  �I��?޸R@�\)BR�
C{�I��=�\)@�G�Bg�HC2�R                                    Bx]���  �          @�  ���@4z�@�  Bi(�B����?W
=@��
B���C.                                    Bx]�f  m          @�G���(�@o\)@�
=BT{B�8R��(�?�ff@��B�.B�p�                                    Bx]�  
�          @θR��z�?p��@�=qB��RC����z�5@��B�.CJ��                                    Bx]��  
�          @�G���33����@���B:Q�C~�쿓33��p�@:�HAޣ�C��                                    Bx]�-X  �          @�=q�G��~�R@��RBE
=C�q�G����@H��A�Q�C�Ф                                    Bx]�;�  
�          @�=q������\@�B@33CxR�����z�@E�A�z�C�t{                                    Bx]�J�  ;          @�{��ff�U�@�33BcG�C|� ��ff���@}p�B
=C��3                                    Bx]�YJ  
�          @�p�����P  @��Bg(�C|O\������H@���B{C��
                                    Bx]�g�  T          @�������=q@���B�
=Cl�Ϳ����l��@��HBRp�C|�                                    Bx]�v�  
�          @�ff>��A�@�z�Bu��C�G�>����R@��HB*{C��                                    Bx]ȅ<  �          @θR�����B�\@���Bv  C�Ff������
=@�33B*Q�C��
                                    Bx]ȓ�  
�          @�Q�Y��� ��@\B��Cv�Y���xQ�@�33BP\)C���                                    Bx]Ȣ�  
�          @�\)��{��=q@���B��{CW���{�N{@�G�B[z�Cp                                      Bx]ȱ.  "          @�ff�{�c�
@�\)B�\)CI���{�1�@��
BaCgc�                                    Bx]ȿ�  
�          @θR��Ϳc�
@��B�CI������1�@�(�Bbz�Cg��                                    Bx]��z  
Z          @�  �#33���R@�{B�B�C;��#33�{@���Bj33C]�                                    Bx]��   "          @����G����H@�Q�B�L�CV�{��G��@  @��B]�Co�f                                    Bx]���  T          @�ff���Ϳ�p�@ƸRB�Q�C^޸�����Y��@���B\�RCt�
                                    Bx]��l  "          @�p�����@�
=B�W
CAk����\)@�
=Bo�
CdǮ                                    Bx]�	  
�          @�=q�u��G�@�ffB�p�Cp���u�#�
@ƸRB���C�H�                                    Bx]��  �          A#\)��\)�s33A!G�B��RC}h���\)��(�A�
B�ǮC��                                    Bx]�&^  �          A$�׿5@�\A��B��B�aH�5�z�HA ��B�W
Cj�                                    Bx]�5  "          A1�>�@b�\A'33B��B��>�<�A0��B��{AE                                    Bx]�C�  "          A733?�@�G�A(��B�p�B��q?�?5A5�B��
BK�H                                    Bx]�RP  
�          A3\)�����"�\A$��B�(�Cjc׿�����A33BaQ�C|33                                    Bx]�`�  "          A7���Q��G�A.{B���CfͿ�Q����A�HBoz�C|�                                     Bx]�o�  
�          A6ff?��\@�(�@��B=\)B�k�?��\@tz�A�B�ffB�#�                                    Bx]�~B  "          A3�
@g
=A
=@�
=B��B�u�@g
=@���A\)BT��Be��                                    Bx]Ɍ�  �          A-�?��@���A  B`G�B��R?��@0  A"{B��B���                                    Bx]ɛ�  "          A&�R�L�Ϳ�(�A��B��{CL���L������A	Bf��Ci�R                                    Bx]ɪ4  "          A+
=?��
@�
=AG�B��fB�\)?��
?�  A&�RB��
A��                                    Bx]ɸ�  T          A,Q�=�Q�?���A)G�B�ffB��==�Q���\A&=qB�{C�)                                    Bx]�ǀ  T          A,z�>��@���@���BoB�\>��?�G�A�B�33B�ff                                    Bx]��&  T          A,  ����@��@�Q�B6�
B�L;���@��A�\BG�B���                                    Bx]���  	�          A+���R@�p�A��Bk��B�(���R@A$z�B�z�B��f                                    Bx]��r  
�          A,�׿��@333A  B�\)B��)��녾W
=A�\B��=C;Q�                                    Bx]�  T          A-��@G�A@�  A�(�B�p�@G�@�=q@�Q�B"�B���                                    Bx]��  
�          A-?�AQ�@��RB  B��q?�@ϮA{BO�B��                                     Bx]�d  "          A-?
=q@�ff@���B>33B�z�?
=q@�p�A�B��)B�#�                                    Bx]�.
  �          A0  ?�33@�{@�  B3�B�  ?�33@�A�
Bz�B�                                    Bx]�<�  	          A/�<�@�  A�BJ�\B�� <�@y��AffB�
=B�#�                                    Bx]�KV  m          A/\)��R@�ffA ��B?(�Bמ���R@���A��B��{B�\                                    Bx]�Y�  �          A0��� ��@�ffA�
BRG�B�#�� ��@N�RA#\)B�Q�B�                                    Bx]�h�  �          A4Q��5?��
A-�B��HC�R�5��A+33B��\CX8R                                    Bx]�wH  
�          A3��p��?�
=A0��B�\)B��H�p���   A/33B�.Ct��                                    Bx]ʅ�  "          A2ff���H?�ffA.{B�=qC
���H�33A*�HB���CrE                                    Bx]ʔ�  
A          A0  @Q�A\)@��HA��B�@Q�@�@�\)B-�B�                                    Bx]ʣ:  ;          A1�@
=A&�R?(��@c33B��@
=A�@q�A�=qB��=                                    Bx]ʱ�  T          A-�@�RAp���p��υB�#�@�RA%�������B��{                                    Bx]���  
Z          A,��?���A$  �333�q�B��=?���A)�>��@�B�=q                                    Bx]��,  T          A,z�@(�A$�ÿ�  �=qB�ff@(�A%�?�@��
B��                                    Bx]���  
�          A(Q�?��A z�@  AJ{B��?��Aff@���A�=qB��H                                    Bx]��x  �          A'33��33A�
@�=qA�=qB�ff��33@��
@�B3��B��                                     Bx]��  �          A&�H�L��A�R@l��A�\)B�Q�L��A{@��B
=B�                                    Bx]�	�  �          A!����AG�@��RB
=qB�����@���@��BM33B�(�                                    Bx]�j  �          A!�����@��
@�B.�HB�\����@�  A�Br=qB�.                                    Bx]�'  �          A!���
=@���@�B.�HBƮ��
=@���A�
Brz�BϨ�                                    Bx]�5�  �          A
=�%�@�A�HB�B�C	�f�%���=qA  B�C9�                                    Bx]�D\  �          A (��C�
@�A�B���C���C�
�333AG�B�C@ٚ                                    Bx]�S  S          A33��G�?��A  B�aHC�׿�G����A�B�=qCf�\                                    Bx]�a�  �          A
=�G
=��p�A�\B�.CT� �G
=���\A�HB_��Ck�
                                    Bx]�pN  T          A
=�H���>�RA{Bz�C_���H������@�z�BD��Co^�                                    Bx]�~�  
B          A��<#�
@�ff?��A"�HB���<#�
@�\)@\��A�33B�Ǯ                                    Bx]ˍ�  
�          A+\)@-p�A��q����B�� @-p�A$�ͿO\)���
B��                                     Bx]˜@  �          A+�@P  AG��Tz���p�B�u�@P  A"{�������B�aH                                    Bx]˪�  n          A+\)@tz�Aff�z��IG�B���@tz�A�\?z�@G�B���                                    Bx]˹�  l          A)�@�(�AQ��s33���B��
@�(�A����
����B�
=                                    Bx]��2  �          A+33@1G�A(������Q�B��R@1G�A!���H��p�B�aH                                    Bx]���  �          Aff    >��R@�G�B��B�{    ��ff@�=qB�p�C���                                    Bx]��~  �          A(����R��(�Az�B�G�C[O\���R�l(�A ��B~��Cx�                                    Bx]��$  �          A�
�>{?���A�\B�8RC
=�>{��Q�A�\B�=qCI��                                    Bx]��  T          A����@�\)@�\)BI��B�.���@L(�A�
B��3B�Ǯ                                    Bx]�p  �          A�
�XQ�@��\@�\)Ba��C��XQ�?��A��B���C.                                    Bx]�   
�          A*=q�~�R@�33@��HB =qB�z��~�R@��A	G�BW(�CB�                                    Bx]�.�  �          A.{�g�A=q@��B��B�3�g�@�z�@��B=B��                                    Bx]�=b  �          A.�H�ǮA@��A�G�B��Ǯ@�ff@��\B733B˨�                                    Bx]�L  T          A1��@  A��@�=qA�ffB�33�@  @�\A�B@p�B��R                                    Bx]�Z�  �          A/��\A�@�A�RB�=q�\@�z�@�z�B1�Bʊ=                                    Bx]�iT  "          A)��n{@��H@���B/Q�BĞ��n{@��@�Bo�B���                                    Bx]�w�  �          A"ff��33?�\@�(�B<�
C$�R��33��Q�@�\BC�RC7(�                                    Bx]̆�  <          A���ff@@�  B
=C"\��ff?.{@���B�C-��                                    Bx]̕F  
�          Ap��{@���@p�Ah��CJ=�{@b�\@e�A�  C�                                    Bx]̣�  �          A Q��z�B�\?���@��RC5��z���?�(�@��HC7:�                                    Bx]̲�  "          A ����@#�
@@��A�33C$L���?У�@g
=A�(�C)޸                                    Bx]��8  
�          A{���@�
=?�G�@�
=C�����@�ff@,(�A�ffCn                                    Bx]���  
�          A   ��\)@˅�   �{
=Ch���\)@�녿&ff����Cz�                                    Bx]�ބ  "          A"�H��@���33���B��R��A�R�-p��w�B�33                                    Bx]��*  T          A�R� z�@���?Q�@�\)C��� z�@��@
�HAS
=C+�                                    Bx]���  "          A�\�	@��H?���A7
=C@ �	@h��@C�
A��C{                                    Bx]�
v  
�          A\)��z�@��@G�A��CL���z�@���@�z�A���C�q                                    Bx]�  n          A!���
=@�(�?���A (�C	h���
=@�Q�@HQ�A�p�C8R                                    Bx]�'�  
�          A&�\��Q�@��H@�G�B3z�C����Q�@:�HA=qBY\)C��                                    Bx]�6h  T          A#33���(��@z�HA�  C8{���ٙ�@eA�(�C>T{                                    Bx]�E  T          A#\)�녿�(�@xQ�A���C;n���{@Z=qA��CAW
                                    Bx]�S�  �          A&=q��׾�ff@�p�A�(�C6����׿�  @��
A�
=C>��                                    Bx]�bZ  �          A%���ʏ\@1G�@�\B>��CT{�ʏ\?#�
A ��BN�RC.8R                                    Bx]�q   �          A&{��@u�@���BC��@
=@�ffB#C$#�                                    Bx]��  T          A'��
=>k�?���A��C2�3�
=�.{?��HA��C4�R                                    Bx]͎L  
�          A&�H�p�?�\)���
��{C*+��p�@*=q�b�\��ffC$�                                    Bx]͜�  �          A%G�����@)����  �
=C!k�����@������\��\)CG�                                    Bx]ͫ�  T          A!G���H?�����Q���
=C*����H@)����
=��z�C"��                                    Bx]ͺ>  �          A$����G�@��H�*=q��z�C�3��G�@�33��z���p�C0�                                    Bx]���  
(          A#���\)A z�L�;�\)B�����\)@���@�
A;\)C �                                     Bx]�׊  �          A"�\��G�@��
?�Q�A8��B��H��G�@�\@{�A�B��=                                    Bx]��0  T          A#��Z=q@�(�@��HA��HB�R�Z=q@�
=@�RB6  B�k�                                    Bx]���  
�          A&=q��RA�\?
=@U�B�\��RA�\@I��A��B�z�                                    Bx]�|  �          A%���\(�A�@{AHQ�B�B��\(�A�@��\Aԣ�B��                                    Bx]�"  
�          A-��%@�@�p�B+�B��H�%@��
A{BcQ�B�                                     Bx]� �  T          A.{��\@��A33BD\)B�G���\@��RA�HB|�\B�
=                                    Bx]�/n  
(          A0���   @�33AB�33B�k��   ?�z�A*=qB��C@                                     Bx]�>  
�          A3����@j�HA'�
B��B�z῱�?\(�A1G�B�33CG�                                    Bx]�L�  
�          A3\)�z�H@G
=A+33B��B��H�z�H>�=qA2=qB���C$��                                    Bx]�[`  
Z          A3�>��
?@  A2{B���B���>��
��A/33B��)C�J=                                    Bx]�j  �          A4z�>B�\?��HA1G�B��HB�#�>B�\��z�A1G�B�ffC�ٚ                                    Bx]�x�  
�          A7��#�
@Q�A3�B���B��{�#�
�
=qA733B�aHC}�3                                    Bx]·R  �          A7�
�L��@   A333B���B���L�;�
=A7�B�aHCt��                                    Bx]Ε�  
�          A6�H?��?k�A5p�B�.Bl�?�Ϳ�(�A3\)B�W
C��R                                    Bx]Τ�  
B          A6{=�?��HA4��B��3B�{=���A3�
B��C��)                                    Bx]γD            A6{=�Q�?fffA4��B��)B���=�Q���HA2�RB�
=C�aH                                    Bx]���  �          A6=q>8Q�?L��A5G�B�B�Ǯ>8Q���
A2�RB�
=C�y�                                    Bx]�А  �          A2�H    @���A��Ba  B���    @fffA$Q�B�W
B���                                    Bx]��6  �          A3�>.{@|(�A$��B��B�{>.{?�G�A/�B�ǮB��                                    Bx]���  T          A2=q���R@��\A�\Bq�
B�B����R@1G�A*�HB�8RB�L�                                    Bx]���  T          A0�ÿ��H@�33A"�RB�aHB��Ὶ�H?��HA-�B�#�C�f                                    Bx]�(  	�          A1��Ǯ@g
=A%G�B�L�B♚�Ǯ?uA.ffB���CL�                                    Bx]��  �          A1��p�?�=qA-G�B��B�Ǯ��p��aG�A/33B�G�CWaH                                    Bx]�(t  �          A/�
��  >�A.�RB�k�C,Y���  �"�\A)�B�.Cxp�                                    Bx]�7  "          A0(���?�\A.{B�=qC !H���	��A*�HB��
Cl��                                    Bx]�E�  
�          A0�׿�z�?\(�A-�B��C���z��ffA+33B��Cg�                                    Bx]�Tf  "          A-녿�z�=�A*�\B���C0uÿ�z��{A%�B�� Ch0�                                    Bx]�c  	`          A.{�^�R?��A+�
B�aHB�{�^�R����A+�
B�=qClǮ                                    Bx]�q�  
�          A-p�>�33@-p�A&{B��=B�33>�33>�A+�B�  A���                                    Bx]πX  
(          A+33�#�
@*=qA$��B�{B�8R�#�
=�G�A*{B���B�ff                                    Bx]ώ�  
(          A,(�?L��@eA ��B�ǮB��
?L��?�=qA)B��BVG�                                    Bx]ϝ�  �          A*�H?�z�@�(�A��B�k�B�Q�?�z�?�(�A%B��Bm��                                    Bx]ϬJ  
�          A*=q?E�@���A�\Bn�B�u�?E�@>�RA"�\B�L�B���                                    Bx]Ϻ�  	�          A)�?�ff@w
=A�
B�ffB���?�ff?�Q�A&{B��BW�                                    Bx]�ɖ  
�          A-�?��@��\A�HBt��B�L�?��@{A$��B�p�BR\)                                    Bx]��<  
�          A.{?�Q�@�(�A33BV�B��)?�Q�@z=qAffB�\)B~z�                                    Bx]���  
�          A.ff?�z�@�\)Ap�Bl{B�#�?�z�@H��A%B�(�B��                                    Bx]���  T          A,��?���@��A�BX�RB��q?���@|��A�RB�u�B��                                    Bx]�.  
�          A,Q�?�=q@��
AQ�BQ��B�{?�=q@��RAQ�B��{B�                                    Bx]��  
@          A,z�?��@׮AffBE\)B�?��@�p�A�
BxG�B�                                      Bx]�!z  T          A.�H@�@�ffAffBAQ�B�@�@�(�A��BsB��f                                    Bx]�0   "          A/33?��R@��
A	G�BP�\B�\)?��R@�
=A��B�.B��{                                    Bx]�>�  "          A.�R?�p�@���AQ�BX�B�?�p�@w�A�RB���B{�R                                    Bx]�Ml  	�          A.=q?c�
@�\)A(�B
=B��q?c�
@�A)G�B��B��                                    Bx]�\  "          A.=q<#�
@_\)A#�B�Q�B�Ǯ<#�
?��A,  B��B�B�                                    Bx]�j�  �          A,�ÿJ=q@'�A"�\B��
Bսq�J=q>uA'�
B��C#�                                    Bx]�y^  �          A,(����?�  A&{B�aHC
�׿�녿���A%p�B�Q�Cfc�                                    Bx]Ј  �          A&�R��  >���A"�HB�B�C*5ÿ�  � ��A�
B�.Cd��                                    Bx]Ж�  
�          A%�c�
?��A"ffB�u�B�\)�c�
���
A"�HB�Ce
                                    Bx]ХP  
Z          A)>�\)@�\A$(�B���B��>�\)��\)A((�B��
C�&f                                    Bx]г�  T          A)p��L��?�G�A&�\B��
C s3�L�Ϳ��A&{B��HCo
=                                    Bx]�  T          A(Q���ÿ\A!��B��3CTxR����n�RA(�B�\CmaH                                    Bx]��B  
�          A(����
>k�A%�B�Q�C-� ��
��
A!�B��HCa{                                    Bx]���  
�          A)G��^�R?s33A&�\B��fC�3�^�R����A%�B��Cm��                                    Bx]��  �          A'�
�+��uA'\)B���C9!H�+��
=A#
=B��fC~(�                                    Bx]��4  	�          A)�����>��
A(��B��=C xR����   A%�B��C�^�                                    Bx]��  �          A*�H?��
>���A(��B�  A�=q?��
��z�A%�B�Q�C�+�                                    Bx]��  T          A,(�����@�HA&{B��3B�����=���A*�\B�p�C�                                    Bx]�)&  "          A)��>#�
@EA!B��B��>#�
?Q�A(��B��fB�\                                    Bx]�7�  
�          A)��?�?��HA$(�B�u�BW�R?����RA'
=B��C��
                                    Bx]�Fr  
�          A(��@33@N�RA�B�k�Bf�@33?��
A$z�B�ǮA��
                                    Bx]�U  
�          A0(�@E@��A�
By\)BS{@E?��A&{B���A�Q�                                    Bx]�c�  
�          A%���  A   ?�\)A'�B�{��  A�@~{A���B�8R                                    Bx]�rd  "          A"ff��(�A�>�\)?˅B��쿼(�A�H@��A\Q�B�Q�                                    Bx]с
  �          A>�
=A�\��p��*�\B��R>�
=A��>��?���B���                                    Bx]я�  �          A&=q@��@��
@ٙ�B2  B=q@��@%@��BK33A�                                    Bx]ўV  n          A\)��Q�A33@L��A�33B�.��Q�AG�@�Q�A���B�L�                                    Bx]Ѭ�  
�          Aff?G�A��@���A�G�B�G�?G�@�  @���B��B��q                                    Bx]ѻ�  
Z          A!�?�A��@,��Ax��B�#�?�A(�@�z�A�G�B�=q                                    Bx]��H  
Z          A%p�?�p�A��@�p�A�
=B�W
?�p�@��
@�ffBB�8R                                    Bx]���  "          A%p�@Q�A  @���BB��@Q�@�p�@�B2  B��
                                    Bx]��  
�          A'�@
�HA
=@���A���B��@
�H@���@�\)B(�\B�G�                                    Bx]��:  <          A'\)@Q�A�
@�
=A�
=B���@Q�@�
=@��B&��B�33                                    Bx]��  
          A!p�@�G����R@�BT{C�˅@�G��E@�RBB�RC��
                                    Bx]��  �          A�H@��
@���@���B��A�p�@��
@H��@�p�BG�A�Q�                                    Bx]�",  
�          A�@��H@��@�{B��B'�@��H@n�R@��HB1B��                                    Bx]�0�  
�          A!�@�@��
@�  B'BM�
@�@�(�@���BJffB,p�                                    Bx]�?x  T          A"{@�Q�@�@X��A��HBXz�@�Q�@�33@�
=A�
=BJ\)                                    Bx]�N  
�          A%G�@�z�A
=>�{?�33B|�H@�z�A�H@(�ADQ�Bz33                                    Bx]�\�  
�          A&�H@xQ�@�\)@ʏ\B��Bw{@xQ�@��@���B>p�Ba                                      Bx]�kj  
Z          A$Q�@'
=@6ffA33B�aHB=�H@'
=?s33AG�B���A�                                      Bx]�z  T          A (�?�@���A\)BrB���?�@,��A
=B���B_{                                    Bx]҈�  "          A   ?�z�@�\)A33B^�RB�#�?�z�@o\)A{B��\B���                                    Bx]җ\  
�          A z�@	��@�(�ABY�B�\)@	��@j=qAQ�B��=BnG�                                    Bx]Ҧ  
�          A%��@�H@�G�A{BX�HB��
@�H@q�A��B��{Be�                                    Bx]Ҵ�  "          A((�@:�H@��A
=qB_��Bo��@:�H@P��A�B�\)B@��                                    Bx]��N  
�          A(��@g
=@���A��Bb��BM��@g
=@.{A  B�(�B�H                                    Bx]���            A'\)@XQ�@�\)A	�B`
=BY@XQ�@=p�A�B���B$                                    Bx]���  �          A$Q�@�=q@z=q@��HBUz�B/�\@�=q@�A\)Bp�A�                                    Bx]��@  �          A(  @�ff@Mp�A	G�Ba�
B	ff@�ff?��RA��Bu��A��H                                    Bx]���            A*=q@陚@�G�@
=A\(�A�\)@陚@���@<(�A�G�A�z�                                    Bx]��  "          A/\)A�@����G����A�Q�A�@���Z�H��{A�ff                                    Bx]�2  
�          A/�A\)@����!��W
=B
��A\)@�zῪ=q����B�                                    Bx]�)�  
�          A/\)A��@׮<#�
=#�
B(�A��@��
?��@ٙ�B{                                    Bx]�8~  J          A2=qA=q@�
=?��A	�B�
A=q@���@2�\Ak33B\)                                    Bx]�G$  
�          A5�A�H@�  ?��
A��A��HA�H@��@2�\Ae�A�
=                                    Bx]�U�  �          A4��Ap�@�
=@fffA���A��Ap�@^{@��\A�=qA�G�                                    Bx]�dp  
�          A5��A   @s�
@l��A���A��A   @C33@�33A��HA��                                    Bx]�s  �          A733A(  @I��@L(�A�A���A(  @   @n{A�(�AVff                                    Bx]Ӂ�  �          A7\)A(  @�
@~{A���A1�A(  ?��@���A�Q�@�Q�                                    Bx]Ӑb  T          A8(�A*�H��z�@�\)A���C�8RA*�H����@�33A�\)C�%                                    Bx]ӟ  �          A5�A'
=��G�@�z�A�C�:�A'
=��ff@�(�A�Q�C��                                    Bx]ӭ�  
�          A4Q�A!���@�  A�C�3A!�\)@�p�A���C��                                     Bx]ӼT  �          A3�
A$Q쿴z�@�G�A��C��A$Q��
�H@}p�A�p�C�
=                                    Bx]���  T          A1�A(�Ϳ�@/\)AeC�%A(����@�AE�C��=                                    Bx]�٠  
�          A1�A/
=�˅?��\@�Q�C��qA/
=��  ?5@k�C�u�                                    Bx]��F  �          A0z�A'
=�33@(Q�A\��C��=A'
=�.�R@
�HA5C���                                    Bx]���  �          A/\)A#����@P��A�z�C�RA#��,(�@4z�An�HC���                                    Bx]��  T          A.ffAz��*=q@|(�A��C�c�Az��Tz�@Y��A��C��q                                    Bx]�8  "          A.�RA���U@Z=qA�C��{A���x��@0��Aj�\C�'�                                    Bx]�"�  "          A,��A(��k�@���A��RC�'�A(����@[�A�{C�e                                    Bx]�1�  
�          A)A	���fff@�  A�C��fA	����@�G�A�33C�c�                                    Bx]�@*  
�          A(��A��g
=@�  A�p�C�Y�A���
=@���A�(�C���                                    Bx]�N�  �          A(  A\)�H��@��A��C���A\)����@�
=A�  C�E                                    Bx]�]v  �          A(Q�@�
=�C�
@�G�B(�C�� @�
=���@��A�p�C�|)                                    Bx]�l  |          A#�
@�p��A�@�\B0(�C��=@�p����R@�ffB
=C��)                                    Bx]�z�  "          A!�@�p��p  @陚B8�RC��
@�p���ff@�G�B!(�C��                                    Bx]ԉh  �          A#
=@��s�
@���B9�C��@���\)@�(�B ��C�\)                                    Bx]Ԙ  "          A%p�@��R��ffA��Bw�HC���@��R�%A  Bi�RC��{                                    Bx]Ԧ�  �          A%�@��R    AQ�Bt�C���@��R��ffA=qBn�HC�W
                                    Bx]ԵZ  T          A%p�@�����A(�Bg(�C��@���1�A�\BY=qC�G�                                    Bx]��   �          A%p�@�\)�z�HA�BWC�S3@�\)�Q�A z�BM33C�&f                                    Bx]�Ҧ  T          A$z�@��
��
=A	p�Bb�HC��q@��
��{A=qB[33C��
                                    Bx]��L  "          A%�@�33����@�G�BE=qC�J=@�33���@�B;ffC��                                    Bx]���  "          A%�@�ff��@�{B){C��\@�ff�3�
@��B
=C��                                    Bx]���  o          A&ff@�z��
=q@߮B(��C��
@�z��S33@���B�C���                                    Bx]�>  �          A'�@����{@�ffB5��C���@���4z�@�\B*�C�<)                                    Bx]��  T          A$(�@߮����@�=qB7(�C���@߮�!�@߮B-  C��                                    Bx]�*�  "          A#33@�(���z�A�Btz�C�G�@�(���  A33Bl��C�(�                                    Bx]�90  T          A"=q@�G�?�A{Bv
=@�p�@�G��p��Ap�BtffC�p�                                    Bx]�G�  T          A ��@^{?�
=A��B�A�G�@^{����AffB�z�C�(�                                    Bx]�V|  T          A33?�\)?޸RAp�B��B<33?�\)>#�
A�
B�  @���                                    Bx]�e"  T          Aff?�  ?��A{B��)B?�  �8Q�A�B�z�C�                                    Bx]�s�  
�          A   @^�R?�=qA�B�
=AÅ@^�R=��
A�B�=q?��\                                    Bx]Ղn  9          A33@|��?У�A�B��A�33@|��>.{A{B�(�@{                                    Bx]Ց  
�          A�@�(�?��AB{p�A�  @�(�>�ffA��B���@�{                                    Bx]՟�  �          A�@��R?��
A�Bz�A�\)@��R>�33A��B��f@���                                    Bx]ծ`  �          A=q@l��?�A  B�p�A�{@l��>k�AffB��H@`��                                    Bx]ս  �          A$  @��H?�
=A�\B~��Aə�@��H>�A��B�k�@�
=                                    Bx]�ˬ  �          A'
=@{��A!B��fC��
@{�ffAffB��C��\                                    Bx]��R  
�          A(  @@  =�Q�A ��B���?�33@@  ��p�A33B��)C�ٚ                                    Bx]���  T          A'�@/\)>��RA!��B�z�@��@/\)��G�A z�B��C��H                                    Bx]���  T          A#
=@G�>\A=qB�A=q@G���z�AG�B��C��=                                    Bx]�D  "          A#33?�(��W
=A�
B��=C��?�(��޸RAp�B�
=C�AH                                    Bx]��  
�          A#33@(�?O\)A�\B��A��@(��5A�RB��fC��
                                    Bx]�#�  �          A ��@#�
?�33Az�B��A�z�@#�
��\)A=qB�.C�33                                    Bx]�26  T          A ��@=q?@  A�RB�ǮA�z�@=q�8Q�A�RB��)C��H                                    Bx]�@�  
�          A Q�@R�\�.{A�HB�C�0�@R�\�A�B��C���                                    Bx]�O�  "          A ��@~�R��\A33B��HC�T{@~�R���AQ�B�HC�N                                    Bx]�^(  
�          A ��@h�ÿc�
Az�B�
=C�)@h���G�A��B�B�C�f                                    Bx]�l�  
�          A   @)���k�A��B�G�C�c�@)����A��B�z�C�L�                                    Bx]�{t  "          A ��@S�
�^�RAB��HC��)@S�
�  A{B��3C��                                    Bx]֊  "          A�R@H�ÿJ=qAp�B���C���@H���
=qA�B��3C��H                                    Bx]֘�  �          A�@6ff��z�A��B���C�)@6ff��Q�A�\B�33C���                                    Bx]֧f  
�          A z�@�þ.{A�B�B�C���@�ÿ˅Ap�B���C�:�                                    Bx]ֶ  T          A�@1�<��
A  B�ff>���@1녿�{A�\B��{C��                                    Bx]�Ĳ  �          A=q@2�\�ǮA�B��)C�f@2�\��G�A��B�  C��=                                    Bx]��X  "          Aff@333��RA\)B�=qC��H@333��p�AQ�B�ffC�ff                                    Bx]���  �          Aff@L�Ϳ��Ap�B��{C���@L�Ϳ�
=A�\B���C�o\                                    Bx]��  �          Aff@@  �^�RA{B�33C���@@  ���A�\B��
C���                                    Bx]��J  T          A�R@6ff=�G�A�RB�L�@��@6ff��p�Ap�B�=qC�U�                                    Bx]��  T          Aff?ٙ���
=A=qB�ǮC���?ٙ��!�A{B���C��R                                    Bx]��  
�          A�\@��G�AB�C��{@��Q�A�\B��C�+�                                    Bx]�+<  
u          Ap�?\���AB�8RC�Q�?\�*�HA�B��=C���                                    Bx]�9�            A��?�{���RAp�B�=qC���?�{�3�
A��B��\C���                                    Bx]�H�  �          A=q?�Q���
A�B�=qC��?�Q��EA�
B�B�C�]q                                    Bx]�W.  �          AG�?�
=��\)A�B�p�C�n?�
=�+�A��B�aHC�3                                    Bx]�e�  �          A�\?��
���\A=qB��C�,�?��
�#33A{B��C��                                     Bx]�tz  T          @ָR@{�@  @���B�p�C���@{�Ǯ@��
B�B�C�o\                                    Bx]׃   �          @�G����
@p���z�Q�BĊ=���
@L�����ǮB�z�                                    Bx]ב�  �          Aff���@i�����H�w�B��쿱�@�������ZffBը�                                    Bx]נl  "          A�
>�  @J�H�
ff��B�\>�  @���ff�x\)B�k�                                    Bx]ׯ  �          Aff>L��@�(����
�<�B�8R>L��@޸R����{B���                                    Bx]׽�  
�          A�׿��@�����H�,G�B��ÿ��@�{��z��{Bǳ3                                    Bx]��^  T          A���ff@�33�ə��*��B�p���ff@�(����
�
=B�\)                                    Bx]��  
�          A{����@����=q�/p�BԳ3����@��H���(�B�B�                                    Bx]��  
(          A�H�=q@�  ���&{B�33�=q@�
=�����	��B�#�                                    Bx]��P  "          A
�\��
=@e��{�0G�C&f��
=@��H��p���C
�H                                    Bx]��            Az��ff@�G���Q��8��B�33�ff@��H��ff���B�B�                                    Bx]��  
�          A��*�H@�Q������   B����*�H@�
=��Q��33Bܙ�                                    Bx]�$B  
�          A
=���R@�����{�i��B�����R@����=q�Np�Bݳ3                                    Bx]�2�  
�          A{�ff@�{��p���  Bҏ\�ffA33�X�����HB�                                    Bx]�A�  
�          A�\�\Ap��~�R����BȨ��\Az��8Q����BǨ�                                    Bx]�P4  �          A�R��\)A��W�����B��ÿ�\)A����f�\B�{                                    Bx]�^�  �          A\)�
=qA=q�E��33B��)�
=qA�� ���Ip�B���                                    Bx]�m�  
�          A=q��(�A\)�2�\��z�B�  ��(�A�
�޸R�4Q�B��                                    Bx]�|&  
�          A(����A
=�*=q���BȽq���A\)��=q�"ffB��                                    Bx]؊�  
�          A�
���\A  �+���ffB����\AQ�˅�#33Bą                                    Bx]ؙr  T          Ap��fffA�\�U���(�B�(��fffA(�����f�RB��                                    Bx]ب  �          A{�&ffA�H�%��ffB���&ffA
=���
��B��\                                    Bx]ض�  
�          A
�H��33A	��fff��\)B��3��33A	>#�
?�=qB��                                    Bx]��d  
�          A   ����@�z�@�A�z�B��Ϳ���@��@G
=A��
B�Ǯ                                    Bx]��
  "          @�=q?E�@���@�Q�BS�HB�� ?E�@b�\@ǮBoQ�B�aH                                    Bx]��  �          @�\?s33@(�@߮B�B�B�8R?s33?�=q@�ffB�u�BZ                                      Bx]��V  T          @�z῵�   @��
B�(�CGs3����{@�Q�B��qC_��                                    Bx]���  "          @�=q�����J=q@�B��CUn������
=@���B�aHCju�                                    Bx]��  
�          @�
=�:�H>�@�B�  C���:�H��p�@��
B��fCO�                                    Bx]�H  
�          @�33>\)�L��@�ffB�C�>\)���H@�B�u�C�`                                     Bx]�+�  �          A  ?�  ?�z�A�B��\B,��?�  >�G�A�B�� A��                                    Bx]�:�  
Z          AG�?���?�RA�
B��A�\)?��þ��A(�B��C�c�                                    Bx]�I:  T          AQ�?h�ÿc�
A
�RB�B�C�Ǯ?h�ÿ�33A(�B�#�C�˅                                    Bx]�W�  
Z          A
�R?8Q�:�HA	p�B�Q�C�H�?8Q��p�A33B���C�XR                                    Bx]�f�  �          Ap�?8Q쿱�A
=B��C���?8Q���HA�B��3C�B�                                    Bx]�u,  
�          A	�?!G���G�A�B�
=C��?!G���p�A��B��C��                                     Bx]ك�  
�          A�>8Q쿋�A=qB�  C���>8Q��33A\)B��C���                                    Bx]ْx  
�          A  >W
=���A�RB�G�C�B�>W
=�ffA�B�G�C���                                    Bx]١  �          A  >�\)���
A�HB�� C��H>�\)���RA  B��3C��                                    Bx]ٯ�  
�          A��?J=q���A{B�W
C�|)?J=q�  A�HB��C��H                                    Bx]پj  �          AQ�?�G���ffA��B���C���?�G���RAffB��C�1�                                    Bx]��  	�          A\)�\)�%�A z�B��C����\)�]p�@�{B�
=C�o\                                    Bx]�۶  
�          A{�&ff�%@�B��qC�H�&ff�]p�@��HB�8RC���                                    Bx]��\  �          Aff�k��[�@�\B��qC~��k���  @���Bj�RC�޸                                   Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�$�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�3�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�B@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�P�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�_�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�n2              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�|�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]ڋ~              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]ښ$              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]ڨ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]ڷp              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�Լ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��b              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]� �              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�T              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�;F              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�I�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�X�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�g8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�u�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]ۄ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]ۓ*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]ۡ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]۰v              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]ۿ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��h              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�%�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�4L              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�B�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�Q�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�`>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�n�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�}�  L          A�H@�
��z�@�(�B9G�C��{@�
�θR@\B&�C��
                                    Bx]܌0  
          A�@����
=@ٙ�B>��C�Y�@���ə�@ȣ�B+�RC�e                                    Bx]ܚ�  T          A=q@p����@޸RBG�C�#�@p���  @�ffB5
=C�!H                                    Bx]ܩ|  
�          A=q@>{��{A=qB���C��q@>{�%A33B���C�y�                                    Bx]ܸ"  �          A=q@C�
���A  B�L�C��\@C�
��A��B��=C�:�                                    Bx]���  "          A{@
=q��G�A
�HB�ffC�h�@
=q� ��A�
B�ffC�]q                                    Bx]��n  
�          A�@��(�A
=qB�.C�K�@�p�A\)B�.C�0�                                    Bx]��  
�          AQ�@
=�'
=AQ�B��fC�@
=�Tz�A Q�B|\)C���                                    Bx]��  �          AQ�@\)�Mp�AB��C�p�@\)�x��@��BpffC��                                    Bx]�`  "          A  ?s33�w�A ��B~�
C��f?s33��G�@�ffBk�RC��                                    Bx]�  �          AQ�?�p��n�RA��B�L�C�  ?�p�����@�Q�Bm��C���                                    Bx]��  T          AQ�?������@���B\G�C��?�����@�{BIG�C�w
                                    Bx]�-R  �          Az�?�����(�@��HBY�RC�8R?�����\)@�(�BF��C���                                    Bx]�;�  
�          A  ?�Q���  @�ffBT��C�e?�Q����H@�\)BB  C��f                                    Bx]�J�  "          A  ?������@�
=BKQ�C��?���ƸR@�
=B8G�C���                                    Bx]�YD  �          A  ?�G���(�@�Q�Bd�C��?�G����@�33BQ�HC�                                    Bx]�g�  �          A@�=q?���@��Bq��A^{@�=q>��@�p�Bu(�@�
=                                    Bx]�v�  �          Az�@�Q�?G�A�RBxQ�A'
=@�Q�=�G�A\)BzG�?�(�                                    Bx]݅6  �          A(�@���\A(�B}�C�` @����ffA33Bz�RC��\                                    Bx]ݓ�  �          AG�@~�R��A�HB�� C��f@~�R�8Q�AffB���C��H                                    Bx]ݢ�  �          Az�@N�R�L��A
{B�ǮC�f@N�R��  A��B��C���                                    Bx]ݱ(  �          A  @Dz��z�A�
B��)C��@Dz��%A��B�ffC��=                                    Bx]ݿ�  "          Az�@Q���\A�B���C��=@Q��<��A�\Bvp�C���                                    Bx]��t  "          AQ�@J�H�ffA�HB�aHC�E@J�H�1G�A�B{��C�w
                                    Bx]��  T          A�@<(��XQ�@�
=Br��C��H@<(���Q�@�BdQ�C�#�                                    Bx]���  T          A�@>�R�5�A{B|C�9�@>�R�^{@�(�Bo�C�S3                                    Bx]��f  �          A��@\(��(��@��Bv{C�C�@\(��P��@�  BjQ�C�E                                    Bx]�	  "          A@R�\�G�A�\B��C���@R�\�9��@�ffBtz�C�K�                                    Bx]��  
�          A@j=q��33A�B�\C��@j=q��\AG�Bz�C�s3                                    Bx]�&X  >          A��@a녿�A�B���C�O\@a��33@��RBx�C�n                                    Bx]�4�  j          A��@o\)���
Ap�B}�C��f@o\)�	��@�ffBv{C��                                    Bx]�C�  �          AG�@u��k�A
=B���C�=q@u���ffAp�B{�RC��R                                    Bx]�RJ  �          Aff@n{�c�
Az�B�ffC�L�@n{�\A�HB\)C��H                                    Bx]�`�  �          A=q@s33��Q�A�\B~{C���@s33��
A z�Bv�RC���                                    Bx]�o�  
�          A�@�G��E�@��
Bn�\C�1�@�G���\)@�G�Bj�\C���                                    Bx]�~<  T          A33@��\���
@�\)Bpz�C��@��\�!G�@�ffBoG�C�)                                    Bx]ތ�  �          A{@}p��\)A�HB�
C���@}p���
=AB|�C���                                    Bx]ޛ�  
�          Az�@`�׿��A\)B�  C��=@`�׿�Ap�B~C���                                    Bx]ު.  
�          AQ�@k�����A�B�z�C��f@k����H@��By�HC��R                                    Bx]޸�  �          A��@e�Tz�A�
B��)C�|)@e��Q�A�\B�G�C��                                    Bx]��z  �          A�\@]p����A\)B�#�C�!H@]p�����AffB�=qC�.                                    Bx]��   "          AG�@X�þ�\)AffB��fC���@X�ÿh��AB�k�C��H                                    Bx]���  �          Az�@U�L��Ap�B�(�C�=q@U��A(�B�� C��H                                    Bx]��l  T          AQ�@^�R�
=A��B��HC�7
@^�R����A�B�ǮC���                                    Bx]�  
�          AG�@xQ쾙��A�\B�  C��3@xQ�c�
A�Bp�C��                                    Bx]��  "          A��@Z=q�:�HAp�B��\C��)@Z=q���A(�B�#�C�P�                                    Bx]�^  �          A(�@N{��ffA=qB���C��@N{���AG�B��3C��                                     Bx]�.  T          A�R@s33�=p�A   B�Q�C�}q@s33����@�p�B|G�C�n                                    Bx]�<�  "          Aff@�p��Y��@��Bu�C�8R@�p���z�@�
=BqQ�C���                                    Bx]�KP  
�          Ap�@�(�����@��HBnC�ff@�(��	��@�ffBg��C�G�                                    Bx]�Y�  �          A@�G���ff@�G�Bw=qC���@�G�����@�{Br�C�+�                                    Bx]�h�  �          A�@�
=�:�H@�\)Bnz�C�]q@�
=���
@��Bj�HC��                                    Bx]�wB  �          A�@�=q�}p�@���Bo��C���@�=q���
@��Bk{C�<)                                    Bx]߅�  "          A{@�����@�ffBqC��@���˅@�33Bl��C��)                                    Bx]ߔ�  T          A�@�Q��\)@�\BU��C�f@�Q���@�ffBO�C���                                    Bx]ߣ4  
�          A@��ͿǮ@��HB`��C��q@����z�@�RB[{C���                                    Bx]߱�  �          A=q@��\����@�G�Bu��C�Ǯ@��\��  @�{Bp�C�e                                    Bx]���  �          A33@�G���(�@�BiQ�C�~�@�G���  @�Q�BdQ�C�y�                                    Bx]��&  
�          Aff@��\����@�{Bo��C��@��\��{@�33Bj�C��\                                    Bx]���  8          A�H@��ÿaG�@�=qB\z�C��@��ÿ��@�BX��C�J=                                    Bx]��r  p          A(�@��H�!G�@�\)BUp�C��)@��H����@��BR�
C�H                                    Bx]��            A��@��
=@�
=BSz�C��@���=q@��BQ
=C�]q                                    Bx]�	�  "          A@����@�Q�BQ�C���@����ff@�ffBOQ�C�s3                                    Bx]�d  �          A
{@��\)@�BKz�C�Ff@����@��BJffC�޸                                    Bx]�'
  T          A�@�  �aG�@���Ba��C��H@�  ����@�ffB^  C���                                    Bx]�5�  T          A{@�Q쿆ff@�p�BbC���@�Q�Ǯ@�\B^��C��3                                    Bx]�DV  T          Aff@��
��G�@��BP�C��@��
��p�@�{BM  C�G�                                    Bx]�R�  �          A�H@�  ��ff@��BZ�\C��3@�  ���@�p�BV  C�"�                                    Bx]�a�  "          A�\@�33���R@��
B_=qC�Ф@�33�޸R@��BZ�C�%                                    Bx]�pH  	�          A��@������@�z�BZ(�C���@����,(�@�\)BS(�C�Q�                                    Bx]�~�  	�          AG�@�z���R@�
=B]�RC�/\@�z��.{@陚BVz�C���                                    Bx]���  �          AG�@����@�z�Be��C��)@�����@���Ba  C�33                                    Bx]��:  
�          A(�@�����@���B]C�R@�����@�G�BX�C���                                    Bx]��  
�          Ap�@�p���@�z�Bg(�C��)@�p��$z�@�B_�RC�/\                                    Bx]ๆ  T          AQ�@|���8Q�@�\)B_�RC��\@|���W
=@���BVQ�C���                                    Bx]��,  
�          A
=����=q@��RBz�Cw�\����33@��A��RCx�                                     Bx]���  
�          A	p��z��ə�@�G�BG�C{�
�z���(�@��B�
C|�3                                    Bx]��x  "          A	p���  �ָR@�ffB=qC��
��  ��G�@�  BffC��{                                    Bx]��  �          A	��������\@���B=p�C~�\�����\)@��B.�C��                                    Bx]��  �          A�Ϳ�z�����@�z�B8C�N��z����@�  B*
=C���                                    Bx]�j  T          A(�������=q@�ffBQG�C���������@˅BB��C�~�                                    Bx]�   
�          A��W
=���H@��Bb
=C��R�W
=��G�@�  BS\)C�K�                                    Bx]�.�  
�          A  �Q���{@�\Bo33C�}q�Q����@ᙚB`�\C��                                    Bx]�=\  T          A�;B�\��  @�=qBoG�C��R�B�\��
=@���B`p�C��R                                    Bx]�L  T          A	p�?8Q��n�R@��B~z�C�|)?8Q���
=@��Bo�C��q                                    Bx]�Z�  �          Az�z��P��@��\B�(�C��3�z��p��@�33B}�RC��)                                    Bx]�iN  �          A�R?.{���@ƸRB>�C�xR?.{���@��HB0=qC�>�                                    Bx]�w�  "          Azῢ�\��Q�@��B�\)CC�f���\�Y��@��
B�=qCU��                                    Bx]ᆚ  �          Az�c�
�~�R@�(�Bp�HC����c�
��@ۅBb�C�XR                                    Bx]�@  �          A����@��B]p�C�1�����@�  BN�
C�l�                                    Bx]��  �          A  ?�\)��
=@˅BB��C��=?�\)���@��B4p�C�h�                                    Bx]᲌  
�          Az�?˅�
�H@��B��)C�*=?˅�'�@�Q�B���C��=                                    Bx]��2  �          A�
@  ?�(�A�
B�{B��@  ?���AG�B�A�Q�                                    Bx]���  T          A�@4z�@(�@�p�B�p�B\)@4z�?�Q�A ��B��=A��H                                    Bx]��~  
�          A
=@%�@S33@�G�Bs�RBO��@%�@3�
@�\)B~��B=�                                    Bx]��$  	.          A�?(��?�33AffB��B��?(��?^�RA�B�W
BT{                                    Bx]���  
�          A�W
=?n{A�
B���B�G��W
=>�p�Az�B�k�B��                                    Bx]�
p  
�          AG�>�p�@z�A(�B�(�B���>�p�?��
A
=qB�=qB�k�                                    Bx]�  
�          AQ�>�@)��Ap�B�=qB�(�>�@�A�
B�L�B�\)                                    Bx]�'�  T          A33?�Q�@'
=A
=B�p�B��?�Q�@Ap�B��
BqG�                                    Bx]�6b  
�          A�?n{?\AQ�B�Q�Bi�H?n{?}p�A	��B�
=B:=q                                    Bx]�E  T          A
=?}p�@�
A�B��B��=?}p�?��
A�
B�Bd=q                                    Bx]�S�  T          Az�G���(�A\)B�Q�CP���G��xQ�A
�RB���Cg
                                    Bx]�bT  T          A녿(�?8Q�A��B��RC�(�>8Q�A�B���C#��                                    Bx]�p�  	�          A
=���H?�A
=B��BШ����H?�33A��B�W
B���                                    Bx]��  "          A(��B�\?���Az�B�u�B���B�\?��A
{B��BĔ{                                    Bx]�F  �          AG�>#�
@z�A(�B�ffB�\)>#�
?��A
=qB�z�B��                                    Bx]��  
Z          A  �u@c33@�\)B�B��ͽu@C33A�HB��B��                                    Bx]⫒  �          AQ쿘Q�@tz�@�\)Bwp�B֙���Q�@Vff@��RB�p�B��                                    Bx]�8  
�          A�R�h��@z�A ��B���B�(��h��?���A�\B��B��
                                    Bx]���  
�          Aff�h�þ��
A��B�CG��h�ÿTz�A(�B�
=C^��                                    Bx]�ׄ  �          A녿�=q?�{A z�B�\)CxR��=q?\(�A��B��
C(�                                    Bx]��*  �          A33�
=@5@��B��RB˙��
=@ffA�B���B�\)                                    Bx]���  "          A\)����@S33@��
B�p�B�aH����@4z�@��B���B�G�                                    Bx]�v  �          A
=���@\(�@��HB�=qBݽq���@>{@���B��qB�                                    Bx]�  
�          AG���{@e@�p�B|�\B�G���{@HQ�@�(�B���B���                                    Bx]� �  "          A����@
=@�
=B���B�G����?��@�33B���B���                                    Bx]�/h  "          A�ÿ�\)?�\@��RB�p�Cp���\)?��
A ��B�=qC�                                    Bx]�>  "          A���n{@=p�@��B�\)B��n{@�R@��B�  B�G�                                    Bx]�L�  T          AQ�Q�@J�H@�\B�=qB��Q�@,��@�Q�B���Bճ3                                    Bx]�[Z  
�          A=q���@Y��@�RB�B������@<(�@��B���B���                                    Bx]�j   
�          A�H�n{@��@�p�Bg��B̨��n{@s�
@��BuG�B�W
                                    Bx]�x�  
�          Ap����H@��@�G�B^��B�k����H@y��@�G�Bl
=Bօ                                    Bx]�L  T          @陚�
=q@~{@�33Bb�B�z�
=q@e@ʏ\BpB��                                    Bx]��  �          @���!�@���@�p�B��B����!�@��@�\)B,\)B�3                                    Bx]㤘  
�          @���   @���@VffA�33B���   @��@k�B G�B�aH                                    Bx]�>  T          @�p���
=@���@n{B��B��
��
=@�=q@�Q�B�RB��H                                    Bx]���  "          @��?G�@[�@ƸRBq�
B�8R?G�@B�\@���Bp�B��                                    Bx]�Њ  "          @�ff?\(�@Z=q@��HB{(�B��\?\(�@?\)@���B�Q�B��q                                    Bx]��0  T          @�>u@���@�G�BQz�B�.>u@���@�=qB_p�B���                                    Bx]���  
�          @�z��Y��@dz���(��,�RC���Y��@vff�����"ffCp�                                    Bx]��|  T          @�z��g�@J�H��p��<{C
� �g�@^�R���R�2��C�                                    Bx]�"  �          @��
��33@P  �|���Q�C(���33@^�R�o\)� Q�CJ=                                    Bx]��  �          @�����@�R����
=C�����@�R�����
Q�C�
                                    Bx]�(n  
�          @��H��(�@'
=��ff��HC
=��(�@8Q���G��G�C��                                    Bx]�7            @�\��z�@��ff�6�C�q��z�@=q��=q�0ffC�)                                    Bx]�E�  
�          @�(��Q�>�=q�љ��}=qC/:��Q�?+��У��{=qC(k�                                    Bx]�T`  "          @��R����@W�����2��C������@l(���{�)��Cc�                                    Bx]�c  �          @����=q@x�������,�RCG���=q@�ff��G��"�HC�                                    Bx]�q�  �          @�\)��z�@�z���33�6{Bޏ\��z�@��R��G��)  B�.                                    Bx]�R  "          @��H�333@���p��B�.�333@�{�����B�                                    Bx]��  �          @���o\)@�G���G��z�Cu��o\)@����  ��\B��3                                    Bx]䝞  �          @���=q@{��Tz���
=C:���=q@��
�Dz��ə�C޸                                    Bx]�D  "          @�ff����@���'�����C
�=����@�����Q�C	�{                                    Bx]��  q          @�p��)��@�=q�=p���33B����)��@���'����
B�z�                                    Bx]�ɐ  �          @����\@�ff�<(����HB�33���\@��
�"�\��\)Bɣ�                                    Bx]��6  "          @��Ϳh��@�(��1G���=qB�33�h��@�G��Q���ffB���                                    Bx]���  �          @�ff���@�ff�0����\)B�����@˅����\)B��)                                    Bx]���  
�          @�\�8Q�@�p��e����HB���8Q�@�(��N{�ѮB�B�                                    Bx]�(  
Z          @�z����\@�(���\)�33C�R���\@�(��}p��CaH                                    Bx]��  �          @�{�p��@}p���(��!{C���p��@����(���RC�{                                    Bx]�!t  
w          @���s�
@u��{�p�CǮ�s�
@����{�G�C޸                                    Bx]�0  
=          @ָR�\@�33����<{B�z�\@�������.  B��
                                    Bx]�>�  
�          @��@=p�@�z���(��/(�BY��@=p�@�����#�\B`�                                    Bx]�Mf  "          @�\@\(�@S�
��
=�>G�B/�R@\(�@hQ���  �433B:(�                                    Bx]�\  "          @�G�@|��@J=q���
�5G�B�@|��@^{����,(�B%33                                    Bx]�j�  �          @��@P��@q�����7�RBD@P��@��H����,��BM�                                    Bx]�yX  
�          @�  @{�?.{�����^  AG�@{�?����  �Z��Al��                                    Bx]��  
�          @�{@�zᾳ33��p��9��C��
@�z�����:�C��
                                    Bx]喤  
�          @陚@�H>�����\)�@�  @�H?5��ff��A��                                    Bx]�J  
�          @��H�L��?�  �陚�HB�33�L��@����p���B��                                    Bx]��  �          @�����?����  �)B�=���?�\)��z�.B�z�                                    Bx]�  T          @�=q��33>Ǯ����¬\)C��33?^�R���¦33Bߊ=                                    Bx]��<  T          @�G��#�
@0������nG�C��#�
@I����\)�cz�C#�                                    Bx]���  �          @ᙚ�vff@'�?޸RA��C���vff@   ?�33A��C�q                                    Bx]��  �          @�R�$z�?��@�=qB��C���$z�?k�@�z�B���C c�                                    Bx]��.  
w          @�����R@�ff@^�RA��HC����R@~{@o\)Bz�C��                                    Bx]��  
�          @�z����\@��\?�\)A��CY����\@�  ?�33A6{C�
                                    Bx]�z  "          @����(�@�녿��
�
�\B���(�@��
�0�����B�33                                    Bx]�)   
�          @�\�޸R@�ff��  �B�.�޸R@�ff>.{?�z�B�.                                    Bx]�7�  �          @�33�u@7
=�����y��B�8R�u@N{���\�kB�Q�                                    Bx]�Fl  
�          @�z��   @����AG���{B�Ǯ�   @�ff�*�H���RBڞ�                                    Bx]�U  �          @�Q���@/\)@�p�B�RB�L;��@�@��\B�B�33                                    Bx]�c�  �          @���z�@w
=@���B\�B��׾�z�@`  @�Q�Bj��B��=                                    Bx]�r^  T          @��H>�p�@��
@�=qBU�RB���>�p�@p  @�=qBdG�B���                                    Bx]�  �          @�p���@�33@��\BJ=qB�  ��@�  @�33BXB�8R                                    Bx]揪  
�          @�Q�#�
@�{@��\BC(�B�#׾#�
@��@�33BQ��B�z�                                    Bx]�P  T          @���@���@�(�B0Q�B��)��@�
=@�B?
=B�
=                                    Bx]��  
�          @�33���
@�{@�  B5z�B�\���
@���@���BD33B�33                                    Bx]滜  �          @�{��\)@�=q@�  B2(�B��콏\)@���@�G�B@��B��                                    Bx]��B  �          @�{�L��@���@�p�B$33B��f�L��@�Q�@�
=B3  B�33                                    Bx]���  "          @�(����@��@�B1��B�Ǯ���@u�@�{B@G�B��
                                    Bx]��  �          @�
=�=q@�p�@�RA��B�L��=q@���@ ��A��B�                                    Bx]��4  T          @Å?W
=@�G�@N{B�B��?W
=@�=q@a�B��B�.                                    Bx]��  �          @��>.{@�z�@G�A��RB�\>.{@�\)@,��A���B���                                    Bx]��  
�          @�R?333@�33@-p�A��B��?333@��@H��A�p�B���                                    Bx]�"&  
x          @���?��@��H@\)B�\B���?��@��@�33Bz�B�                                    Bx]�0�            @�  ?��@���@#�
A�p�B��?��@�
=@=p�A��HB�=q                                    Bx]�?r  �          @�z�>�z�@�=q@
=A�G�B�W
>�z�@�p�@\)A�p�B�33                                    Bx]�N  �          @�
=?.{@�33?G�@�\)B���?.{@�G�?���A+�
B��3                                    Bx]�\�  "          @Ϯ�O\)@�Q�?��A�ffB�k��O\)@��
@�A��\B�Ǯ                                    Bx]�kd  
�          @�p��1�@��?��\A9G�B�(��1�@�
=?��Ao�B���                                    Bx]�z
  �          @���x��@�  =�?��B��x��@�\)>�ff@��\B���                                    Bx]爰  "          @���p�@p�׿O\)��
=C����p�@s�
�\)��\)CE                                    Bx]�V  �          @������@1G���G��MG�C������@6ff�����.=qCB�                                    Bx]��  
Z          @�
=�-p�@�ff?���A~{B� �-p�@��\?�
=A�  B�                                    Bx]索  
�          @�\)��@�=q?�33A�33B����@�@  A�{B��{                                    Bx]��H  
Z          @����2�\@�
=?p��A33B�#��2�\@���?�  APz�B��                                    Bx]���  
�          @��R��G�@�\)?��
Ar{B�aH��G�@�(�?�=qA�z�B�(�                                    Bx]���  
F          @�{��@�����Q��,G�B�(���@����mp���B���                                    Bx]��:  6          @�Q쿈��@o\)��Q��_�B�
=����@�(�����P��B�.                                    Bx]���  T          @�녾k�@�=q�����O��B�#׾k�@�{��
=�?��B���                                    Bx]��  �          @�zᾣ�
@�z���G��6��B�33���
@����p��'=qB��q                                    Bx]�,  �          @�R��@�(�����"B��=��@�ff���R�
=B���                                    Bx]�)�  
�          @�ff���@�����\�(�B��
���@�(���(��\)B�Q�                                    Bx]�8x  
�          A�\���@�����
�133B����@ə���ff�!�\B�p�                                    Bx]�G  T          Aff�}p�@�(���  �HG�B�녿}p�@����(��8��B�L�                                    Bx]�U�  �          A�\����@�  ��33�L��B�#׿���@�����<��B�G�                                    Bx]�dj  T          A�ÿk�@�  ��p��a��B�{�k�@�\)���H�Q�
B��                                    Bx]�s  T          A�xQ�@qG���z��x�
Bг3�xQ�@������i33B�ff                                    Bx]聶  �          A�Ϳ�  @S33���fBոR��  @tz���G��v=qB�Q�                                    Bx]�\  "          Aff�^�R@e��G��p�B�B��^�R@��
��Q��o��B��f                                    Bx]�  
�          A\)�W
=@]p���p�=qB�#׿W
=@�  ����t��B˙�                                    Bx]譨  T          A\)�E�@h����33��RB���E�@�p���=q�o�RB�                                      Bx]�N  
�          A�R�W
=@�Q���
=�k��B�B��W
=@�Q���z��[�
B���                                    Bx]���  
�          A��L��@z�H��\)�wffB���L��@�ff���g=qB�Q�                                    Bx]�ٚ  �          Ap��@  @y����\�v33Bɽq�@  @�p������e��B�=q                                    Bx]��@  �          AQ�O\)@�=q��  �gz�B�\)�O\)@����p��W(�B�33                                    Bx]���  �          A��O\)@�Q��أ��Y�HB�33�O\)@�������Ip�B�z�                                    Bx]��  T          A녾��H@��R��ff�n{B�aH���H@�
=��(��]�B��                                    Bx]�2  "          AQ�#�
@���޸R�e��B�z�#�
@������
�U
=B���                                    Bx]�"�  T          A(��0��@���\�lffB��)�0��@�{��  �[��B��)                                    Bx]�1~  �          A��Q�@�������mffB�B��Q�@�p��ڏ\�\��B��)                                    Bx]�@$  "          A�R�E�@�����z��[�HB�\)�E�@�Q��У��K(�BĨ�                                    Bx]�N�  	�          Aff�Q�@��R����e�B��f�Q�@�
=�ָR�TffB���                                    Bx]�]p  �          A�c�
@�z���ff�n�B�W
�c�
@�p����
�]\)Bɨ�                                    Bx]�l  T          A�R��ff@�p������`��BθR��ff@����{�P�B��                                    Bx]�z�  �          A=q�k�@��H��\)�k�B�ff�k�@�33�����[{Bʨ�                                    Bx]�b  �          A{�\(�@}p������p�B�\)�\(�@�\)�ָR�_33Bɞ�                                    Bx]�  r          A
=�Q�@l(���Q��y��B���Q�@�\)�޸R�i  B���                                    Bx]馮  h          A�
�333@[���R�qB���333@~�R���rffB��
                                    Bx]�T  �          A�\��=q@b�\����|�B��쿊=q@��H��\)�kQ�Bх                                    Bx]���  
�          A33��  @%��ffQ�B�LͿ�  @J�H��\)
=B�{                                    Bx]�Ҡ  �          @�  ��33@��������`ffB�\)��33@�Q������N�HB�p�                                    Bx]��F  "          @����@x����Q��h�\B�z��@����ff�V�HB��                                    Bx]���  
�          @��H��Q�@u�����nB��R��Q�@��\�Ǯ�]
=B�ff                                    Bx]���  �          @�ff���
@aG��ڏ\�zB������
@�G������h��B�G�                                    Bx]�8  T          A   ��=q@Tz������)B�.��=q@w���  �s�HB��)                                    Bx]��  
�          @������@N�R��Rk�B�(�����@q����u  B��                                    Bx]�*�  �          @�����
@L(���
=\B�p����
@p  ��ff�v(�B��q                                    Bx]�9*  �          @�(����@HQ���
=\B��\���@l(��޸R�x{B��                                    Bx]�G�  �          A녾���@2�\��(��\B�p�����@X����(�� B�B�                                    Bx]�Vv  �          AQ�aG�@(Q���=q�qB�aH�aG�@O\)���H��B���                                    Bx]�e  
(          A
=?(��@Q���(�ffB�k�?(��@0  ���\B��                                    Bx]�s�  �          A ��?333@ ����Q�W
B��q?333@(Q���=q�=B�B�                                    Bx]�h  "          A ��?fff?�\��G�p�B{�?fff@�����
��B���                                    Bx]�  
F          A�\?��R?�{��z�#�BQ
=?��R@\)���\)Bs�H                                    Bx]ꟴ            @�G�?�  @P  ���
�y�
B�k�?�  @s33�ҏ\�h�B��)                                    Bx]�Z  "          @�z�?�  ?�{��33�fBw��?�  @�R��p�ffB�8R                                    Bx]�   "          @��H?0��?��H���
�qB�L�?0��@���ff�)B�(�                                    Bx]�˦  �          A ��?
=q@
�H���ǮB�#�?
=q@333�����\B�=q                                    Bx]��L  "          @�
=>�33@\)���.B���>�33@G
=��\ǮB��                                    Bx]���  T          A z�?�=q?����\)��Bp�?�=q@!G����B�\)                                    Bx]���  
�          A Q�?=p�@
=q��p��)B�W
?=p�@2�\��R�3B�p�                                    Bx]�>  �          @�Q���@@  ��z�aHB��R���@e�ۅ�y��B�Ǯ                                    Bx]��  "          @���>u@����8RB��>u@B�\��ff��B�{                                    Bx]�#�  	�          @��\��{?��H��� BǊ=��{@%��33�B��
                                    Bx]�20  T          @��
���@
�H��{��B�Q쿅�@1G���\)  B�8R                                    Bx]�@�  �          @��
��?�����33\B��H��@p���p��{B��
                                    Bx]�O|  �          @���E�@
=��G��
B�녿E�@0  ���Hz�B�.                                    Bx]�^"  
�          @��Ϳ0��?������
k�B�LͿ0��?�����z�B�=q                                    Bx]�l�  
�          @���\)?aG���G�Cs3�\)?�  ��z���C\                                    Bx]�{n  
�          @�z��y��?�33�˅�eC#�
�y��?ٙ��Ǯ�^��C}q                                    Bx]�  
�          @����ff@@  ��p�  B�  ��ff@g
=��(��np�B�                                    Bx]똺  
�          A�R�\@J�H����B�#׿\@s33�ᙚ�o�\B߀                                     Bx]�`  "          AG��z�H@aG���ff�|ffB�\�z�H@�z����
�h�
BΔ{                                    Bx]�  
�          @��R��z�@}p��أ��h��B�=q��z�@�G������UffB֊=                                    Bx]�Ĭ  
�          A��@  @ ����Q��B�uÿ@  @J�H��Q��3B���                                    Bx]��R  �          A��=�G�?�ff��z�33B�=q=�G�@   ��ff  B��                                    Bx]���  "          A�\>�p�@������B��>�p�@1G�������B��{                                    Bx]��  �          Ap�?c�
@]p���
=�
B�L�?c�
@��
��(��m��B��q                                    Bx]��D  "          A{>��
@�33��G��rG�B�
=>��
@����z��]��B�B�                                    Bx]��  T          A=q>���@q���p��xp�B�  >���@�p��ٙ��c�RB�G�                                    Bx]��  �          A33@�R@�p�����C
=B��@�R@��R��{�/�HB�z�                                    Bx]�+6  �          @�p�?�p�@�
=��p��M=qB�.?�p�@�Q����R�9�B���                                    Bx]�9�  �          @�{?n{@�{����d�RB�{?n{@�G���  �P{B��                                    Bx]�H�  �          A{?��@|���߮�n�B���?��@��\���H�Z�B��                                    Bx]�W(  
�          @�(�>�33@�����z��0ffB�>�33@�����33�(�B�W
                                    Bx]�e�  T          A33=#�
@��R�ۅ�d
=B�{=#�
@�=q��p��N��B�.                                    Bx]�tt  
�          Ap��Q�@�33����k33Bʅ�Q�@�\)��  �V�Bǔ{                                    Bx]�  T          Aff�z�?ٙ���=q{C���z�@����(�� C                                    Bx]��  �          A{�j=q?����{  C+#��j=q?�  ��33�u��C!.                                    Bx]�f  �          A z�����?�  �Ϯ�X��C$�{����?�{���H�Q�\C��                                    Bx]�  
(          Ap��`��?u��{�|z�C$�f�`��?������t�HC��                                    Bx]콲  �          @���n�R?�����z��o�C���n�R@�\��
=�f��Cc�                                    Bx]��X  
Z          A z��[�?=p������z��C'޸�[�?�����t{C                                      Bx]���  �          @�ff��  �;���
=�0�RCU)��  �=q��\)�<
=CP=q                                    Bx]��  
�          @�z��[����
��(��,=qCik��[����\�����=�RCe�                                    Bx]��J  �          @����E��
=��  �'�\Cn)�E�����R�:G�Ck)                                    Bx]��  �          @�\)�G���\)����CG�CtaH�G����
���WG�Cq+�                                    Bx]��  �          A   �{��p���p��4��Cv��{���H�����I\)Ct0�                                    Bx]�$<  �          A Q�� ����ff�����8�\Cs�� ������Ǯ�L�Cp�                                    Bx]�2�  �          A   �'���(���G��/p�Cr��'����������Cz�Cp�                                    Bx]�A�  �          A�����ff�����<33Ctk������H�˅�P�\CqY�                                    Bx]�P.  �          A ������33��
=�?�Cs�3����\)���S�Cp��                                    Bx]�^�  �          A Q��%����R��(��G�RCp��%��u��љ��[Q�Cl
=                                    Bx]�mz  �          A   �>{��������I
=CjxR�>{�aG���G��[ffCe��                                    Bx]�|   �          A   �AG���z���(��HffCi��AG��`  �У��Z�Ce8R                                    Bx]��  T          @����p��U���
=�i\)Ci���p��(Q�����{p�Cb�R                                    Bx]�l  �          @�(���33�\����G��o�RCq0���33�.�R���)Ck@                                     Bx]��  �          @��R���\�b�\��Q��w33CzLͿ��\�333��\�{Cu��                                    Bx]���  �          @�
=��  �n{�ۅ�n�
Cx{��  �?\)��ff=qCsff                                    Bx]��^  �          A �ÿJ=q�Z�H��\)=qC�t{�J=q�)����C}L�                                    Bx]��  �          A (���G��z=q�ۅ�lQ�C|!H��G��J�H��\)u�CxT{                                    Bx]��  �          @�\)��Q��j=q��\)�t�C{�R��Q��:=q��\�{Cw��                                    Bx]��P  �          @��R���Z=q��B�C������)����33C���                                    Bx]���  �          @��R=�G��  ���
\C�c�=�G���
=���.C�.                                    Bx]��  �          @���?�33�z���p�W
C�9�?�33>k���{��A��                                    Bx]�B  T          @�33>�{��Q������3C���>�{�c�
����¥�C�h�                                    Bx]�+�  �          @�G���33�qG���  �h�Cy����33�C33�ۅ��RCuY�                                    Bx]�:�  �          @���?
=��\����8RC�8R?
=�p����p�£� C��R                                    Bx]�I4  	�          @��H��
=�
=��ff=qC�녾�
=������aHC~��                                    Bx]�W�  �          @���>��
�z���338RC�b�>��
��p�����¡z�C�Ff                                    Bx]�f�  �          @�z�?!G�����=qC��?!G���=q����¡�qC�/\                                    Bx]�u&  �          @��?�p���
=��z�C�H�?�p���{����33C�                                    Bx]��  �          @�\)?�녿��
��\)aHC�S3?�녿333�� �C�G�                                    Bx]�r  �          @���>�{��33��33
=C���>�{�J=q���§=qC��H                                    Bx]�  �          @��\>L�Ϳ�  ���
�fC�T{>L�Ϳc�
��Q�¦�C�s3                                    Bx]  �          @���    �������C��    �xQ���=q¥��C�f                                    Bx]�d  �          @�=q�L�Ϳ�p���=q�qC�B��L�Ϳ�{���£��C��\                                    Bx]��
  �          @�G����������k�C�����}p���
=¥Q�C�Ff                                    Bx]�۰  �          @��<#�
��z���33  C�:�<#�
�
=q���R¬C���                                    Bx]��V  �          @�>u��z���\¢B�C��H>u��z����®ffC�H                                    Bx]���  �          @���>��������£�C��R>���W
=��z�°�C�<)                                    Bx]��  
�          @�z�W
=�u��\¥8RC��þW
=���
��z�°��CI��                                    Bx]�H  �          @��
�
=��G���(�C{��
=�fff����£��Cl��                                    Bx]�$�  �          @�������{��
=C��=���������«�=Cm�                                    Bx]�3�  �          @��Ϳ
=�fff��G�£Cl�3�
=���
���H«(�C5��                                    Bx]�B:  �          @��ÿ�R��
=���Hk�Cr0���R���R��©L�CN��                                    Bx]�P�  �          @��?W
=��R����(�C�` ?W
=��������C��q                                    Bx]�_�  �          @�p�?�  ��������C�H�?�  ��\)�߮C�\                                    Bx]�n,  T          @��
?�Q�?��H�׮��B�\?�Q�@   �љ�u�BGz�                                    Bx]�|�  �          @陚����>�  ��Q�­�\C�R����?�{��¡�3B�Ǯ                                    Bx]�x  �          @��þ�p�>�  ��­
=C�q��p�?�{���¡p�B�ff                                    Bx]�  �          @�׿�  ��\)���£��C7�῀  ?L�����{C��                                    Bx]��  �          @��þǮ���
��«�C[xR�Ǯ?
=q��33©G�B�
=                                    Bx]�j  �          @�{�xQ����¤�)C;LͿxQ�?E���=q ��C��                                    Bx]��  T          @�Q쿮{��33��Q�Q�CB���{?   ��  �{C�f                                    Bx]�Զ  �          @�
=��?�\��ffC8R��@ �����o33CB�                                    Bx]��\  �          @陚���R?!G��ָR��C)���R?����ҏ\  C                                    Bx]��  �          @陚�!G�?�=q��
=�C���!G�?�\)��G��}�CQ�                                    Bx]� �  �          @���>�R?�\��z��v{CE�>�R@#33���
�e\)Cp�                                    Bx]�N  �          @�׿�G�?   ��  #�C33��G�?�\)��(�ǮC}q                                    Bx]��  �          @��ÿ�G��\)�߮��C8���G�?=p��޸RG�CO\                                    Bx]�,�  �          @���p�������=q=qCL�)�p���z����33C:                                    Bx]�;@  �          @�Q��Q녿0����  �w(�C?��Q�=�����G��yC25�                                    Bx]�I�  �          @�\)�,���G
=����[�\Ce{�,���ff�ƸR�p=qC\�q                                    Bx]�X�  �          @�����s33��ff�HCI�������У�\C6�
                                    Bx]�g2  �          @�녿�Q쿵��Q��\C\���Q�z���z��CF޸                                    Bx]�u�  T          @�녿���aG���33G�C;  ���?���ʏ\C!�                                    Bx]��~  T          @�  ��@H������Z
=B�����@s33����@�B��                                    Bx]�$  �          @�=q�)��@�z��
=�qp�B�aH�)��@�녿n{��B�{                                    Bx]��  �          @�녿��?�  ��Q�
=Cn���@33��Q��~�HC8R                                    Bx]�p  �          @�녿�33?���{  B��ΐ33@   ����HB�\)                                    Bx]�  �          @ٙ��0��?�R��p�¤.C
E�0��?��R������B��)                                    Bx]�ͼ  �          @��W
=@{��
=�M�C���W
=@HQ���33�:�C�q                                    Bx]��b  �          @�  ��
=@j�H>�Q�@U�C����
=@e?W
=@��\C(�                                    Bx]��  �          @��
���
@p�׿8Q��ȣ�C  ���
@tz�W
=��{C�\                                    Bx]���  T          @������R?���s�
�"�HC!�R���R?���e��RC�
                                    Bx]�T  �          @�(��.{�!G�����¤� C^�H�.{>�z��ٙ�§��C��                                    Bx]��  �          @޸R���ÿ5��z�¦�{Cu@ ����>B�\��p�­z�C�                                    Bx]�%�  �          @޸R��\)����ָR�)CN5ÿ�\)>�Q���
= �C"(�                                    Bx]�4F  �          @�33?˅�'����Hk�C���?˅��p���z�  C�T{                                    Bx]�B�  T          @�  ?��.�R��p��zG�C��=?������׮�C�33                                    Bx]�Q�  �          @�=q?�{��\����)C�/\?�{�L�����H�C�˅                                    Bx]�`8  �          @�Q�?��ÿ�p����{C�  ?��ÿG���\)��C��q                                    Bx]�n�  �          @�=q@'
=�(�����H�rffC�O\@'
=���H��z�W
C�j=                                    Bx]�}�  �          @�@G��8Q�����`G�C���@G���p������s�HC��3                                    Bx]�*  �          @�@j�H�W
=����BC���@j�H�"�\�\�V�
C��                                    Bx]��  T          @�(�?�(���  ��z�8RC�/\?�(��B�\��=qC�b�                                    Bx]�v  �          @��;��R=�����  ®C!�{���R?������¡z�B�33                                    Bx]�  �          @���8��@$z���{�h  C
W
�8��@Z�H�����P��C0�                                    Bx]���  �          @����z�@���33
=C�
�z�@Q���ff�k�
B�z�                                    Bx]��h  �          @��H�.�R@   �ʏ\�n��C	�=�.�R@XQ�����V��C �R                                    Bx]��  �          @�=q���@G��У��|�C}q���@K���(��d33B��
                                    Bx]��  T          @��QG�@
=��p��_  C)�QG�@L(������I��C��                                    Bx]�Z  
�          @����i��@!G���z��J33C\)�i��@QG���
=�5�\C
�                                    Bx]�   �          @����(Q�@\)��{�sffC�)�(Q�@G
=�����[�C33                                    Bx]��  �          @�R�^�R@E����FC
\)�^�R@w
=��\)�/{C�                                    Bx]�-L  �          @��X��@B�\���R�HC
)�X��@s33��ff�0�
C�                                    Bx]�;�  �          @�{�Y��@G�����@�
C	xR�Y��@u�����(�C}q                                    Bx]�J�  �          @ڏ\�^�R@L(������8=qC	� �^�R@xQ���(�� {C��                                    Bx]�Y>  �          @����e�@O\)���R�*�HC	�{�e�@w��{���
C�=                                    Bx]�g�  �          @�{���@a��j�H�p�C�����@�G��Fff�ݮC�)                                    Bx]�v�  �          @θR�7�@33�����\z�C\)�7�@C�
���
�E
=C33                                    Bx]�0  �          @׮�"�\?��H��(��z  C���"�\@%�����c��C�\                                    Bx]��  �          @�(��"�\?����p�C޸�"�\@.�R�\�i�C޸                                    Bx]�|  �          @�R�p�?��H�ϮW
Cff�p�@:�H���
�h33C\                                    Bx]�"  �          @��
�2�\@33��z��n(�C�=�2�\@L����
=�U�C�                                    Bx]��  �          @�(��W�@&ff��ff�T�HCh��W�@\(���\)�=�Cz�                                    Bx]��n  �          @���s33@fff�����/33C���s33@��\����CO\                                    Bx]��  �          @�  �vff@`  ��G��0p�C	���vff@����ff�\)CE                                    Bx]��  �          @�p���G�@���n�R��Q�C�
��G�@��H�AG���z�C��                                    Bx]��`  �          @�R����@p  �|����C
����@�=q�S�
�܏\C	)                                    Bx]�	  �          @�
=���R@��\�5��33C� ���R@��R��
���C��                                    Bx]��  �          @�33����@�=q�%��33B��
����@��Ϳ�Q��Tz�B��                                    Bx]�&R  �          @�Q���\)@��ÿ�\)�G�B����\)@��R�0����G�B���                                    Bx]�4�  T          @�R���R@Å�333����B�����R@�z�>�=q@B���                                    Bx]�C�  �          @�{���@�{��G���G�B��=���@��ýL�;�
=B�                                    Bx]�RD  �          @�����@�녾��
�   B�����@�G�?&ff@�G�B�.                                    Bx]�`�  �          @�{��  @�
=�\)��
=C�q��  @����У��M�C�                                    Bx]�o�  �          @������\@��\�/\)���RC����\@�ff����m�C                                      Bx]�~6  �          @�\����@�Q��/\)����C&f����@�(���33�q��C�                                    Bx]��  �          @�=q��p�@���?\)��G�C�\��p�@���	������CG�                                    Bx]�  �          @�����@��:�H��=qC���@��H����Q�C}q                                    Bx]�(  �          @�  ����@�������Q�C������@��\�\�B=qC�R                                    Bx]��  �          @�Q���@�G��B�\��Q�C	z���@�\)��\��{C��                                    Bx]��t  �          @��H��p�@�Q��ff�c\)B�����p�@�\)�Tz���  B���                                    Bx]��  �          @����{@��H���H�X(�CO\��{@�G��L����=qC8R                                    Bx]���  �          @�\��
=@��R�xQ���C ���
=@�G��#�
��Q�B�W
                                    Bx]��f  �          @����p�@�  ����pQ�C� ��p�@�  ���\�C#�                                    Bx]�  T          @��H��{@��
��ff�B�HC W
��{@����
=��33B��                                    Bx]��  
�          @�\����@����.{��=qB�B�����@\>�33@-p�B�                                    Bx]�X  �          @�������@���z��r�RB�G�����@�p��n{�陚B�
=                                    Bx]�-�  �          @�R��  @�=q�����N�HC8R��  @�Q�.{��ffC5�                                    Bx]�<�  �          @���{@��׿���+�
C����{@�p���(��Y��Cٚ                                    Bx]�KJ  �          @�����\)@��R��ff�(�C:���\)@�녾����(�C��                                    Bx]�Y�  �          @�����R@�z῜(��B�z����R@��׾u����B�G�                                    Bx]�h�  �          @�p���\)@��Ϳ�\)�1�C���\)@�����(��[�C �H                                    Bx]�w<  �          @�(���Q�@�\)��p��`��C�{��Q�@��R�n{����C
(�                                    Bx]��  �          @����z�@�(����|��C�{��z�@��Ϳ�
=�=qC��                                    Bx]���  �          @�G���ff@�(�������C
u���ff@�\)�����Q�CG�                                    Bx]��.  �          @����\)@�ff�(������C� ��\)@��H�����q�CW
                                    Bx]���  �          @�G���p�@��H�:�H��z�C���p�@�������Q�C W
                                    Bx]��z  �          @ᙚ�s�
@��R�a���C��s�
@�  �)�����B��                                    Bx]��   T          @����G�@���Vff���
C����G�@�(��!���Q�CL�                                    Bx]���  �          @�{���@<���(Q���
=Cs3���@W��z���G�C8R                                    Bx]��l  �          @�����@XQ쿈���Q�C�����@aG���ff�r�\C�\                                    Bx]��  �          @�
=��(�@�{�
=����C� ��(�@�  ��=q�0��C
�q                                    Bx]�	�  �          @������@�{�7��£�C
����@�(������RC޸                                    Bx]�^  �          @���`  ?��H��Q��\��C���`  @=p���=q�D�C��                                    Bx]�'  �          @���|��@=q��Q��B�
C���|��@S�
����*�C��                                    Bx]�5�  �          @�����?��R��ff�2\)C�H���@7
=��Q����C=q                                    Bx]�DP  T          @������@\)������C������@P  �xQ���C��                                    Bx]�R�  �          @��H��  @���#33���RC
=��  @�{��=q�N�HC �                                    Bx]�a�  �          @�  �xQ�@�녿5��Q�B��
�xQ�@�33>��R@(��B��                                     Bx]�pB  �          @�ff��@������G�B�����@��?   @�(�B���                                    Bx]�~�  �          @�Q���{@����=q�S�C �H��{@�{�
=��(�B�(�                                    Bx]���  �          @�  ���@�ff��(����
C �����@�
=�z�H�G�B��\                                    Bx]��4  �          @�Q�����@�{�)����\)C33����@�33�޸R�g�C�\                                    Bx]���  �          @�G�����@�  �{���HC�=����@�(�����K
=C^�                                    Bx]���  �          @�33���\@��R�Vff��Q�C!H���\@�Q��������C�{                                    Bx]��&  �          @����@��H����G�CT{���@���=q�.�\Cc�                                    Bx]���  �          @�����@��R�����{C�{��@�����=q�.�RC�{                                    Bx]��r  �          @����  @���7���=qCT{��  @�=q������ffC��                                    Bx]��  T          @ᙚ��\)@��J=q�ԸRCO\��\)@�ff�����p�C&f                                    Bx]��  T          @�\��  @�
=�.�R���
C����  @��Ϳ��
�iC)                                    Bx]�d  �          @�33���@��)�����C�
���@�33���H�_�C!H                                    Bx]� 
  �          @��H���
@�
=�>{�Ə\C�����
@��R��
��\)C                                      Bx]�.�  �          @�33��p�@���9�����\C���p�@�
=��p���p�C@                                     Bx]�=V  �          @�33��(�@�(��Fff�ϮC�)��(�@���������\Ch�                                    Bx]�K�  �          @���{@��R�9����
=CxR��{@���p���C�\                                    Bx]�Z�  �          @����@�
=�>{���
C�{���@��R�����~�RB�                                      Bx]�iH  �          @�=q�\)@�ff�B�\��=qC ٚ�\)@�ff� �����RB�k�                                    Bx]�w�  �          @�  �S�
@�p��)����G�B�R�S�
@��H����N�RB�
=                                    Bx]���  �          @�ff�w�@����_\)��(�Cff�w�@����!���\)B��                                    Bx]��:  �          @����vff@��R�aG����C��vff@�=q�%���Q�B�ff                                    Bx]���  �          @�����=q@1����
�z�C���=q@b�\�_\)��
=C:�                                    Bx]���  �          @�  ���@)���l(���C�����@U��E��{C�
                                    Bx]��,  T          @�33��p�@J=q������Ch���p�@z�H�\(���p�C
aH                                    Bx]���  �          @�33��33@U�r�\��\C���33@����B�\��\)C
Ǯ                                    Bx]��x  �          @ۅ���R@s�
�b�\����C�=���R@�{�,����C(�                                    Bx]��  �          @�z����@\)�U���(�C	�
���@��\�p���\)C�f                                    Bx]���  �          @��
����@O\)�������C޸����@�G��a����C��                                    Bx]�
j  �          @����Q�@N{��33�'=qCB���Q�@��H�vff�Cz�                                    Bx]�  �          @��H����@3�
��Q��0��C
=����@n{���H�  C	8R                                    Bx]�'�  �          @��H��\)@W�����G�Cs3��\)@����X����  C�\                                    Bx]�6\  �          @�=q����@^�R�����z�C�����@�\)�N{��C:�                                    Bx]�E  �          @��H���@W
=���\�
=C���@�(��S33��C5�                                    Bx]�S�  �          @ٙ����@Vff����
=C�����@��
�U����C��                                    Bx]�bN  �          @�����
=@tz��Y����=qCu���
=@�{�"�\��\)C+�                                    Bx]�p�  �          @�������@p  �W
=��\)C� ����@��� �����C.                                    Bx]��  �          @�(����@i���Vff��ffCh����@�Q�� ����(�C�R                                    Bx]��@  �          @��
����@w
=�P����{C	�{����@��R�Q�����C��                                    Bx]���  �          @Ӆ��p�@dz��W����C���p�@�ff�#33����C�                                     Bx]���  �          @�=q����@����=p���  CB�����@������R��33C��                                    Bx]��2  �          @�G��|��@�(��4z���{C��|��@�(���=q���
C                                    Bx]���  
�          @�G��xQ�@����,(���Q�C���xQ�@�\)���m�B���                                    Bx]��~  �          @љ���z�@xQ��7
=���
C
����z�@��Ϳ�(���ffC�                                    Bx]��$  �          @�ff��=q@g��P  ��p�C����=q@�\)�=q���C	(�                                    Bx]���  �          @�p���Q�@I���qG��	�C���Q�@x���@���֏\CE                                    Bx]�p  �          @�\)���R@C33��Q��z�C�����R@u�QG���Q�CE                                    Bx]�  �          @׮��  @HQ��z�H�ffC#���  @y���J=q��p�C�                                    Bx]� �  �          @Ӆ����@R�\�Vff��C�q����@|(��#�
��Q�CǮ                                    Bx]�/b  �          @Ϯ��(�@XQ��AG��ޏ\C�\��(�@}p��{��=qCp�                                    Bx]�>  T          @�{���H@U�W����HCaH���H@�  �$z���  C	Y�                                    Bx]�L�  �          @�33��\)@W��Q���(�C����\)@�  �{���C�)                                    Bx]�[T  �          @�33��(�@XQ��,(����C޸��(�@x�ÿ������C�q                                    Bx]�i�  �          @�33����@Vff�=p���  Cn����@z�H�	����Q�C\                                    Bx]�x�  �          @�=q����@XQ��333�ҸRC8R����@z�H���R���RC!H                                    Bx]��F  T          @ə����H@`  � ����Q�C�f���H@~{��
=�w�C!H                                    Bx]���  �          @�=q���
@]p��#�
��33C.���
@|(���p��~{C�=                                    Bx]���  �          @�Q����@Mp��5���G�C����@p  �33���HC��                                    Bx]��8  �          @�ff��ff@Tz��0����\)CB���ff@vff�������\C)                                    Bx]���  �          @�Q�����@N{�8Q���G�C�=����@q�����
C{                                    Bx]�Є  �          @�Q���
=@QG��:=q�݅Cٚ��
=@u�ff���HCaH                                    Bx]��*  �          @�Q���
=@R�\�8Q���C����
=@w
=�z�����C0�                                    Bx]���  �          @�  ��ff@E��.�R��33C���ff@g
=���H��z�Cn                                    Bx]��v  �          @���z�@@���/\)���HC\��z�@c33���R��ffC�
                                    Bx]�  �          @ƸR��
=@X���+���=qCٚ��
=@z=q����C
ٚ                                    Bx]��  �          @�\)��@]p��-p���  C�q��@~�R������{C
                                      Bx]�(h  
�          @ƸR���@Y���/\)�ѮC\)���@|(������  C
=q                                    Bx]�7  �          @�p����@Z�H�,�����
C�����@|�Ϳ���G�C	ٚ                                    Bx]�E�  T          @����\@e��#�
��  Ch����\@��\���z�HC�                                     Bx]�TZ  �          @Å��(�@x���{����C�q��(�@�����G��?�C��                                    Bx]�c   �          @�=q��@xQ��z����
C	
��@�Q쿏\)�)�Cu�                                    Bx]�q�  �          @����x��@�=q�
�H���\C���x��@�
=���2=qC
                                    Bx]��L  �          @�p���G�?�z��Tz��=qC33��G�@(���2�\��=qC+�                                    Bx]���  �          @�����
?���U���C#L����
@G��8����  C�                                    Bx]���  �          @�G���=q@�R�J=q��=qC����=q@I���   ���C�)                                    Bx]��>  �          @ȣ���@!��S�
���RC�
��@O\)�'��ŮCT{                                    Bx]���  �          @�  ���R@z��W���HC�q���R@C33�.�R�ϙ�C�                                    Bx]�Ɋ  T          @�Q�����@��^{��
C������@5�8���ۙ�CG�                                    Bx]��0  �          @�  ���
?�=q�^{��RCff���
@&ff�<(���  C�f                                    Bx]���  T          @����\)?��R�u����C:���\)@5�P  ��C
=                                    Bx]��|  �          @��
����?��q��C�R����@,(��O\)��=qC��                                    Bx]�"  �          @����?���xQ��p�C�=���@*�H�U��=qC��                                    Bx]��  �          @�{��(�?���s33���C����(�@)���P����CxR                                    Bx]�!n  �          @�=q��ff?���[���RC���ff@*=q�8Q����HC�R                                    Bx]�0  �          @ȣ�����?��H�L����{C������@*�H�(�����C�                                    Bx]�>�  �          @ʏ\��p�?�
=�\���p�C�{��p�@-p��8Q����C0�                                    Bx]�M`  �          @�(�����?�
=�w���C ������@$z��Vff��\)C��                                    Bx]�\  �          @��H��{?�����  ��\C"����{@Q��aG��=qC)                                    Bx]�j�  �          @ə����?�{��Q��
=C#�R���@33�c33�	ffC�                                    Bx]�yR  �          @�����33?�{���
�$\)C&^���33@�n�R��RC��                                    Bx]���  �          @�  ���R?��
���'�HC$���R@���o\)���C)                                    Bx]���  T          @�  ��ff?��R��ff�)33C$u���ff@�R�qG��z�CY�                                    Bx]��D  T          @�Q����
?��
��33�=(�C%�����
@Q���ff�(C��                                    Bx]���  �          @ə��z�H?!G���(��K�C*�)�z�H?�����=q�:�C)                                    Bx]�  �          @ə��~{?#�
���H�I
=C*�=�~{?��������8
=CW
                                    Bx]��6  �          @����tz�?!G���{�O�C*�)�tz�?���(��>
=CE                                    Bx]���  �          @����u�?���{�O�C+���u�?��
��z��>�HC�                                    Bx]��  �          @ȣ����?\)��{�B  C,B����?�(������2��C^�                                    Bx]��(  �          @����  ?&ff���
�<Q�C+W
��  ?��
����,\)C5�                                    Bx]��  T          @ə�����?333�����E�RC*�����?����ff�4
=C��                                    Bx]�t  �          @�33��33?=p�����7�RC*c���33?�{��\)�'  C�{                                    Bx]�)  �          @��H����?W
=��=q�FffC(.����@����R�2�RC(�                                    Bx]�7�  �          @����\)?k���Q��E
=C&�R�\)@ff���
�0Q�C@                                     Bx]�Ff  �          @ə���G�?�Q���33�<{C ff��G�@#�
���\�!��C�H                                    Bx]�U  �          @�Q��\)?Ǯ�����:C�)�\)@*�H��  ��C5�                                    Bx]�c�  �          @�Q��o\)@����\�<p�C���o\)@HQ��z=q��HC
                                    Bx]�rX  �          @�  �W�@%�����<  C��W�@j�H�o\)���C��                                    Bx]���  �          @�  �:�H@U���(��2�HCE�:�H@��H�XQ���B��
                                    Bx]���  �          @ƸR�fff@G
=�{��(�C=q�fff@�Q��?\)��=qC�                                    Bx]��J  �          @���fff@U�X���
ffC	0��fff@�33�=q����CO\                                    Bx]���  �          @�=q�0  @p���;�� �RB�� �0  @�z��{���
B�=q                                    Bx]���  �          @�{�N{@u�$z���33C���N{@��
��p��tz�B���                                    Bx]��<  �          @����e@Z=q�s�
�=qCxR�e@����2�\��{C��                                    Bx]���  �          @��H�c33@   ���H�:\)C�\�c33@g��q���
Cz�                                    Bx]��  �          @�  �c�
@7
=�z=q�"ffC:��c�
@q��A����C@                                     Bx]��.  �          @��Tz�?��H�����P��C��Tz�@����\�4
=C��                                    Bx]��  �          @��
�[�?���Q��J33CxR�[�@#�
�~�R�+��CB�                                    Bx]�z  �          @���hQ�?�z���  �;Q�C\)�hQ�@.�R�j=q��C\                                    Bx]�"   �          @��R�h��?.{���
�M\)C)z��h��?�\)�����9(�C��                                    Bx]�0�  �          @����8Q�?�����
�i\)C���8Q�@&ff���H�G=qC	�                                    Bx]�?l  �          @�G��Fff?Q�����gz�C%.�Fff@	����ff�M{CG�                                    Bx]�N  �          @\�AG�?aG���ff�k33C#�=�AG�@�R�����O33C��                                    Bx]�\�  �          @�G��7
=?�(���p��k�
C�)�7
=@#�
����I��C
0�                                    Bx]�k^  �          @����*=q?��������uC���*=q@ ������R��C�{                                    Bx]�z  �          @���O\)?޸R��ff�O�RC���O\)@<(�����+Q�C	                                    Bx]���  �          @��H�Dz�?У���\)�\�C���Dz�@:=q����7z�Cs3                                    Bx]��P  �          @\�5?��R���R�m{Cff�5@&ff��{�JG�C	��                                    Bx]���  �          @�(��@��?�G���\)�k  C!� �@��@�������Lz�C�)                                    Bx]���  �          @�ff�B�\?�(���\)�g��C!H�B�\@%��ff�Fz�C��                                    Bx]��B  �          @�\)�(��?��R����x  C��(��@+���{�R�
C�{                                    Bx]���  
�          @�ff�.{?�=q��z��r�C��.{@0  ��=q�M(�C�q                                    Bx]���  �          @���+�?�������tp�C8R�+�@p�����Q��C	s3                                    Bx]��4  �          @�  �'
=?�����R�p��C^��'
=@p����R�L�HC��                                    Bx]���  �          @��$z�?�33��=q�j�HC���$z�@*�H��  �C��C��                                    Bx]��  �          @��ÿ��ÿ=p�����RCJ����?E���33u�C.                                    Bx]�&  �          @�ff�%�?��H��\)�b{C���%�@(Q��y���:�CaH                                    Bx]�)�  �          @�\)�aG�@!G���33�-��Ck��aG�@c�
�O\)���C��                                    Bx]�8r  �          @\�L��?��
����R�C��L��@B�\��(��,Q�Ck�                                    Bx]�G  �          @�
=�<��?�=q�����d�HC���<��@+���ff�@��C	�q                                    Bx]�U�  T          @�  �>�R?�{�����dG�C^��>�R@.{��
=�?��C	�{                                    Bx]�dd  �          @���P��?��R��
=�S(�Cc��P��@0  ����/�C��                                    Bx]�s
  �          @����X��?�  �����G�C+��X��@*�H�s�
�%�\C�=                                    Bx]���  �          @����C�
?�ff��  �\
=C���C�
@%���{�9G�C��                                    Bx]��V  
�          @���<(�?�����ff�fQ�Ch��<(�@����{�D��C.                                    Bx]���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��H   �          @�  �33>�{��(�aHC*��33?������x�C��                                   Bx]���  �          @����=�������3C1L���?�z�����{33C�                                   Bx]�ٔ  �          @��H�h��@9���!����C� �h��@`  ��{���C&f                                   Bx]��:  �          @�{����@*=q�S�
���CG�����@`  ��H��  Cٚ                                   Bx]���  �          @�ff����?�(��e���CW
����@:�H�8Q���C�{                                    Bx]��  �          @�
=����@   �x���#�C������@A��J=q� \)C�q                                    Bx]�,  �          @����e@ff����2�C���e@^{�XQ��p�C�                                    Bx]�"�  �          @����vff@=q�~{�$��C�R�vff@\���Fff��{C
#�                                    Bx]�1x  �          @�(��q�?�Q���p��7�C0��q�@%��dz��{C�3                                    Bx]�@  �          @�G��Z=q?Ǯ�����F  Cn�Z=q@0���p���"�C                                      Bx]�N�  T          @����H�c�
��Q��p�CH5���H?+���G�.C$��                                    Bx]�]j  �          @���z������C?�3�z�?�����\�}��C޸                                    Bx]�l  �          @�33�:�H>�Q�����o\)C,�3�:�H?�������X{C�                                    Bx]�z�  �          @Ǯ�`  ?O\)����Zp�C&��`  @������?p�C:�                                    Bx]��\  �          @�33�s�
?��������8�
Cs3�s�
@A��k��\)C��                                    Bx]��  �          @��
�p��@{�����1��C���p��@W��\(��	  C
!H                                    Bx]���  �          @�
=�x��@��}p��'{Cc��x��@L(��J=q� �C��                                    Bx]��N  �          @����q�?�G���z��?
=C!�
�q�@   �u�� �C�
                                    Bx]���  �          @����e�?�ff����?
=CW
�e�@AG��k����C޸                                    Bx]�Қ  �          @\�n�R    �����O�
C3�R�n�R?�z������C(�CG�                                    Bx]��@  �          @��H�mp�?�����{�J�C#ٚ�mp�@�H����,�C�                                    Bx]���  �          @���s33?�p������B(�C"��s33@!G��|���#��Cp�                                    Bx]���  �          @�(��~�R@�����${C��~�R@[��H������C:�                                    Bx]�2  �          @������@)���qG����C�)���@i���3�
���C
                                      Bx]��  �          @������H@Dz��C�
�C�����H@vff�   ���\C
h�                                    Bx]�*~  �          @�{�\)@+��p  ��
C!H�\)@k��1��ظRC	T{                                    Bx]�9$  �          @�p���ff?�{�.{�ָRC%aH��ff@
=�(���Q�C�                                    Bx]�G�  �          @�{��33?���fff�\)C�f��33@8���7
=��{C��                                    Bx]�Vp  �          @�33���R@!G���H��Q�C����R@HQ��ff�l��C��                                    Bx]�e  �          @��
����@$z��U��C������@]p������C�\                                    Bx]�s�  �          @�Q�����?��H�����(��C����@E�QG��G�Ck�                                    Bx]��b  �          @���P  @��R��G��\(�B���P  @�>��?���B��                                    Bx]��  �          @�  �|(�@����Fff���C5��|(�@�����(��x  CaH                                    Bx]���  �          @�����33@g��A���C
����33@��
���
���\C0�                                    Bx]��T  �          @�{��z�@��� ����C����z�@��Ϳ�R����CQ�                                    Bx]���  T          @�
=���@�Q�������HC
�����@���\(���Cs3                                    Bx]�ˠ  �          @�����\@��\�&ff��  C����\@���p��2�\C
                                    Bx]��F  �          @���\)@���5���\)C��\)@�\)�����RffC&f                                    Bx]���  �          @У��~�R@�\)�Q����C��~�R@��Ϳ���  B�W
                                    Bx]���  �          @�
=���\@����=q��p�C8R���\@�  �����+�C@                                     Bx^ 8  �          @�  ���\@�\)�޸R�y�C����\@��þL�Ϳ�\C                                      Bx^ �  �          @����z=q@��H��Q���=qC ��z=q@�ff��{�A�B��H                                    Bx^ #�  �          @�
=�}p�@�\)��
=����C��}p�@��H��p��P  B��R                                    Bx^ 2*  C          @�{�p  @QG��^{���C
޸�p  @�{�����HC��                                    Bx^ @�  �          @�{�xQ�@33�w��"��CW
�xQ�@X���>{��p�C
ٚ                                    Bx^ Ov  �          @��R����@8���HQ����C&f����@n{�z����C
�                                    Bx^ ^  �          @�����@E��{���C����@l(�����Z�HC��                                    Bx^ l�  "          @����z�@[������  C��z�@vff�8Q����C
Ǯ                                    Bx^ {h  �          @�ff��Q�@^{��R���\C�\��Q�@�  ���
��C��                                    Bx^ �  �          @�(����H@
�H�j=q�33C����H@Mp��2�\��C޸                                    Bx^ ��  �          @��\����?�p��n{���C�{����@C33�:=q��p�C޸                                    Bx^ �Z  �          @�G��s33?�z��z=q�*z�CW
�s33@B�\�G
=�Q�CL�                                    Bx^ �   �          @����j=q?�����=q�B�C#���j=q@Q��p���"�
C�q                                    Bx^ Ħ  �          @��H�Z=q�#�
��ff�W�
C6�3�Z=q?�=q��Q��K�C��                                    Bx^ �L  �          @�=q�AG�������ff�i�
C9��AG�?��
��G��^=qC��                                    Bx^ ��  �          @���{��p���=qk�C?Y���{?�����8RC�3                                    Bx^ �  �          @�  �333�����¦\)C=�q�333?�\��z�ǮB�                                      Bx^ �>  �          @�����=q?�G��r�\� p�C#����=q@���L(��Q�C�H                                    Bx^�  T          @�p�����?�=q�l(��{C#0�����@(��Dz���p�C                                      Bx^�  �          @�����(�?��R�hQ���\C!8R��(�@$z��>{��(�C��                                    Bx^+0  
�          @����  ?�z��l(��{C!�3��  @!G��C33��\)CaH                                    Bx^9�  �          @����z�H��G���\)�<�RC5�f�z�H?�  ��G��2  C"J=                                    Bx^H|  �          @��\���H�.{��z��5p�C6^����H?��~�R�,\)C#�q                                    Bx^W"  �          @����w���{����4�CC�R�w�>����ff�=
=C0+�                                    Bx^e�  �          @����g
=�#33�k��(�CWJ=�g
=��p�����B
=CF��                                    Bx^tn  �          @���z=q��  ����3��CE�=�z=q>\)��G��>�\C1�                                    Bx^�  �          @��H�\)�Q��xQ��.�\C?�\�\)>��H�|(��1�C-�                                    Bx^��  �          @����hQ��  �r�\�&Q�CS�
�hQ�fff����DffCA�                                    Bx^�`  �          @����H���
=����DG�CU��H�ÿz�����a��C>s3                                    Bx^�  T          @�z��333�[��w��$�\Cf�R�333��p������Y�CW=q                                    Bx^��  �          @�
=�
=��  �xQ��"�\Cr8R�
=� ����
=�b\)Cf�                                    Bx^�R  �          @��
�G�����(��iz�C=��G�?�
=�����a��C5�                                    Bx^��  �          @�ff�S�
�����
=�f  C8n�S�
?�(���Q��X��C#�                                    Bx^�  
�          @�ff�j�H=��
��\)�V�C2�R�j�H?��H���D�\C�q                                    Bx^�D  �          @ƸR�aG��u���
�]�C5�aG�?�\)��33�M��C@                                     Bx^�  �          @�ff�Tz�W
=��
=�f{C7�=�Tz�?\��  �WffCO\                                    Bx^�  �          @Ǯ�AG��\)��ff�t
=C6���AG�?�z���{�aC&f                                    Bx^$6  �          @�  �9���W
=�����x�RC8��9��?У������g  C�q                                    Bx^2�  �          @ƸR�Dzᾙ����z��p��C9��Dz�?�  ���bz�C�f                                    Bx^A�  �          @�\)�X��>B�\��{�c\)C0�R�X��?�33���\�L�HC��                                    Bx^P(  �          @ȣ��a�?=p���(��[C(&f�a�@����33�<�\C�q                                    Bx^^�  �          @��H�L(�?�����33�g
=C!��L(�@3�
���>��C
��                                    Bx^mt  �          @�z��'�?�ff�����pffC�
�'�@a�����9ffB�(�                                    Bx^|  T          @����<(�?У���p��h�
C���<(�@Vff��=q�6�CL�                                    Bx^��  T          @����N{?�����p��g�C!���N{@3�
��  �?��C
�H                                    Bx^�f  
�          @��:=q?��������o�C��:=q@J=q�����@{C�3                                    Bx^�  �          @��1G�@33�����e�HC�\�1G�@o\)�����-�B��                                    Bx^��  �          @���&ff@z����H�cp�C
5��&ff@~�R��  �'33B�L�                                    Bx^�X  �          @�(���@y����{�=�
B垸��@�  �Fff��B��                                    Bx^��  �          @�G���@j�H��p��>��B���@����J=q��z�B�q                                    Bx^�  �          @˅���@~{���\�8�B��f���@����>{��\)B�\)                                    Bx^�J  �          @��ÿٙ�@��R����� �RB�Ǯ�ٙ�@�����\���
B�
=                                    Bx^��  �          @ə����
@��������-��B�8R���
@���&ff��(�B��                                    Bx^�  �          @��
����@��
��G��7{B�z����@���8Q����B���                                    Bx^<  �          @��
�p�@Vff�����G��B��q�p�@����Vff��B��                                    Bx^+�  �          @��H�h��?��H��G��DC�H�h��@^{�u��G�CO\                                    Bx^:�  �          @��H�>�R@G����\��C�)�>�R@j=q���&Q�C�                                    Bx^I.  �          @˅�p�@ff���R�mQ�C���p�@u��p��0Q�B�ff                                    Bx^W�  
�          @�33�)��@�����H�e�
C��)��@u��G��*z�B�W
                                    Bx^fz  �          @�=q� ��@�R���H�gffC
aH� ��@z�H��Q��)B�=q                                    Bx^u   �          @˅�p�?�=q����qC���p�@Mp���{�O  B��                                    Bx^��  �          @��   ?����H��C'
�   @p�����]  C�=                                    Bx^�l  �          @��
�'�?p����33�z  C .�'�@.{��ff�LC�H                                    Bx^�  �          @�33�!�?�z������p�C�!�@W����
�8Q�B���                                    Bx^��  T          @����,��@E���{�=G�C8R�,��@�p��G
=��B���                                    Bx^�^  �          @��R�1�@H����
=�4��C���1�@����7���33B�\                                    Bx^�  �          @�  �A�@P����Q��'�HC޸�A�@�ff�(Q��Џ\B��=                                    Bx^۪  �          @�z��5@QG���=q�3�C��5@���:�H��Q�B��                                    Bx^�P  �          @���*�H@o\)�o\)�p�B���*�H@����p���G�B�.                                    Bx^��  �          @��R�J=q@��
��\��  B��3�J=q@�G����R�=p�B�#�                                    Bx^�  �          @�p��R�\@�33�33���B�Q��R�\@��
�&ff�ə�B��                                    Bx^B  �          @��E@�\)�1G���G�B�\)�E@������/�
B�33                                    Bx^$�  �          @�ff�G
=@���0���ݮB��=�G
=@�{��{�,��B�p�                                    Bx^3�  �          @�
=�0  @}p��c33�\)B�� �0  @�ff��Q���p�B�{                                    Bx^B4  �          @��H�*�H@u�_\)��
B��R�*�H@�녿�
=��Q�B��                                    Bx^P�  �          @�{�,��@���W
=�
Q�B�8R�,��@�\)��p���B��                                    Bx^_�  �          @��R��\@vff�����*�B�����\@�����H��=qB�B�                                    Bx^n&  �          @��R�z�@dz���G��8Q�B�L��z�@�33�0����33B�W
                                    Bx^|�  �          @��=q@\)�hQ����B�L��=q@�Q��   ��ffB�aH                                    Bx^�r  2          @����=q@���_\)���B�\)�=q@��ÿ����
B�33                                    Bx^�  �          @��\�3�
@j=q�a����B�{�3�
@�p�� ����=qB�#�                                    Bx^��  �          @�  �0  @�Q��C33� p�B���0  @��\��Q��h��B�\)                                    Bx^�d  �          @�Q��ff@�Q��%��B����ff@�(��Tz����B��                                    Bx^�
  "          @����\)@�Q�����ffB�녿�\)@�Q�������B�z�                                    Bx^԰  �          @��  @�(��@  ��B�{�  @��
��=q�!Bߨ�                                    Bx^�V  T          @ƸR�
�H@��
�  ��  B�  �
�H@�녾L�Ϳ�{B��f                                    Bx^��  �          @�
=�%@����?\)��\B��
�%@��ÿ����#\)B�#�                                    Bx^ �  
�          @Ǯ�*�H@�(��L����p�B��*�H@�
=�����H  B��                                    Bx^H  "          @�
=�1G�@��\�J=q��z�B�G��1G�@�������D��B�=q                                    Bx^�  �          @�G��:�H@�Q��P  ����B���:�H@��
��
=�S33B��                                    Bx^,�  T          @˅�=p�@����z=q���B�aH�=p�@������Q�B�\                                    Bx^;:  "          @ʏ\�G
=@��
�i���  B�
=�G
=@����Q�����B�33                                    Bx^I�  "          @ʏ\�G
=@�=q�mp���B��
�G
=@�(�� �����B�\                                    Bx^X�  
�          @ƸR�>{@����W����B��=�>{@�ff��{�qG�B�u�                                    Bx^g,  
�          @�ff�8Q�@��
�Tz����B��)�8Q�@��׿���eG�B�aH                                    Bx^u�  
|          @��H�4z�@�p��E��=qB�.�4z�@�\)��ff�Ep�B�3                                    Bx^�x  
�          @�33�8��@���L�����B��R�8��@�p���Q��Z�HB�ff                                    Bx^�  T          @����>{@�ff�K���B��=�>{@�녿��H�^�HB���                                    Bx^��  �          @��H�U@g
=�`���(�C�R�U@�z��p����\B�z�                                    Bx^�j  T          @��H�c33@e�Y����RC���c33@�=q��\)��p�B���                                    Bx^�  
(          @�33�G
=@z�H�^�R�\)C k��G
=@����=q���B���                                    Bx^Ͷ  �          @��H�<(�@���P  � B�� �<(�@��
���R�c33B�                                    Bx^�\  T          @��H�=p�@��R�P  ��B�L��=p�@�33��G��ep�B�Q�                                    Bx^�  T          @��H�N�R@r�\�`  ���Ck��N�R@����������B���                                    Bx^��  T          @���O\)@q��]p��33C���O\)@��׿�����z�B�ff                                    Bx^N  �          @�z��HQ�@b�\�_\)��C���HQ�@�녿��H��=qB���                                    Bx^�  
Z          @��H�9��@�  �HQ���RB��H�9��@����(��h��B�\                                    Bx^%�  T          @�G��E@R�\�e��C8R�E@��
�	����
=B��                                    Bx^4@  T          @�Q��*�H@�
=�4z����B���*�H@��R����/�B��                                    Bx^B�  T          @�33�1G�@�\)�!���
=B�p��1G�@��H�8Q����B�\                                    Bx^Q�  �          @�(��H��@�z��0  ��p�B�k��H��@����ff�$��B���                                    Bx^`2  �          @���`��@Mp��S33�
�HC	���`��@�{��33����C��                                    Bx^n�  �          @��H�l��@���z�H�)G�C8R�l��@^�R�5��C�3                                    Bx^}~  �          @�G��hQ�@\)�y���)CT{�hQ�@`���3�
��\C�                                    Bx^�$             @���Z�H@(Q��q��$�HC^��Z�H@u�#33��33C��                                    Bx^��  �          @�ff�l��@���g
=�Q�C#��l��@c33��R��ffC33                                    Bx^�p  D          @�ff�p��@0  �Q��ffC���p��@p������p�C��                                    Bx^�  2          @�33��  @C�
�����C���  @n�R�����6ffC��                                    Bx^Ƽ  �          @��\���
@N{��Q����RC�R���
@mp���R�ʏ\C
                                      Bx^�b  
          @�����@U��R��G�CG�����@{��Y�����C��                                    Bx^�  �          @������@I�����¸RC.���@r�\��  �#
=C�q                                    Bx^�  �          @�G���  @2�\�#33��=qC{��  @b�\��=q�`  C
��                                    Bx^T  D          @�  �r�\@N{�p����C���r�\@s�
�\(��p�C�)                                    Bx^�  
�          @��h��@K�����G�C
�
�h��@vff��=q�7�C\)                                    Bx^�  �          @��R�r�\@hQ쿺�H�x��C=q�r�\@z=q�#�
��G�C
                                    Bx^-F  
|          @�p��r�\@q녿@  ��
=C{�r�\@r�\?5@�Q�C�                                    Bx^;�  �          @���mp�@E��
=���RC:��mp�@e�&ff��p�C��                                    Bx^J�  T          @�G��E@ ���U� {C���E@dz��
�H��G�C�
                                    Bx^Y8  �          @���j=q@C33��  ���C5��j=q@^�R���H��Q�Cs3                                    Bx^g�  
�          @�p��^�R@J�H��
=���C	���^�R@^{����޸RC
=                                    Bx^v�  �          @�
=�l��@E���
=�\��C0��l��@S33=L��?��C
@                                     Bx^�*  D          @�
=�xQ�@6ff�����`z�C���xQ�@E�L�Ϳ(�Cu�                                    Bx^��  2          @�ff�y��@/\)���
�pz�C�R�y��@AG��#�
��z�CL�                                    Bx^�v  T          @�G���G�@0  ��Q��ZffC�R��G�@@  ��\)�O\)Ck�                                    Bx^�  �          @������@'�������{C�����@A녿�\����C�                                    Bx^��  "          @�z��hQ�@	���%��Q�Cff�hQ�@=p�������z�C��                                    Bx^�h  T          @����<(�@   �b�\�*=qC���<(�@h���ff��
=C ��                                    Bx^�  �          @����^�R@
�H�R�\�  C��^�R@P  �\)��p�C	                                    Bx^�  
Z          @�ff�dz�@�R�A��
=C��dz�@L�Ϳ�(����RC
{                                    Bx^�Z  
�          @��R�j�H?���S�
��HC.�j�H@+��\)��C�                                    Bx^	   
�          @�
=�S33@��P  ���C�
�S33@O\)�(��ʸRC��                                    Bx^�  T          @���S33@Q��X���Q�C&f�S33@_\)�  �ȣ�Ch�                                    Bx^&L  �          @�{�O\)?���e�/{C޸�O\)@E�'
=��(�CQ�                                    Bx^4�  T          @�
=�U�@���g��%�RC@ �U�@e�����ԸRC�H                                    Bx^C�  �          @���Z�H@&ff�g
=� 33CǮ�Z�H@qG��
=��33C8R                                    Bx^R>  T          @��R�W�?�Q��u�2z�C#��W�@O\)�4z����RC�                                    Bx^`�  
�          @����AG�@<���?\)�z�C�f�AG�@w
=��z���p�B���                                    Bx^o�  
�          @��
�L(�?����u��8�RC33�L(�@H���6ff�{Cp�                                    Bx^~0  �          @����J=q?�\)��33�J�
C z��J=q@%�W��  C��                                    Bx^��  
�          @�Q��O\)?�\�mp��4\)Cff�O\)@B�\�0�����\C�
                                    Bx^�|  
j          @���Tz�@�R�S33���CT{�Tz�@b�\�
=���
C0�                                    Bx^�"  T          @�  �$z�?��H���R�V�
CL��$z�@K��O\)��C �                                    Bx^��  �          @�{�
=?Ǯ���\�c\)C���
=@E�Z�H�%=qB��                                    Bx^�n  T          @����G�?У��mp��:=qCW
�G�@:�H�3�
�z�C��                                    Bx^�  �          @��R�:�H?��\���
�Q�C�{�:�H@/\)�U���
C�)                                    Bx^�  T          @��H�ff?���=q#�C$��ff@G���Q��Qp�C��                                    Bx^�`  �          @�p��ff?��R��p��k�HCG��ff@5��g��1��C�                                    Bx^	  T          @����(�?������H�p�
Cu��(�@E�l(��0�B���                                    Bx^	�  "          @���33?�33��33�i��CO\�33@Q��g��(ffB�                                    Bx^	R  �          @�
=�33?�����(��ep�C.�33@J=q�\(��%\)B�G�                                    Bx^	-�  �          @�(���
?�\)�����b�\C�{��
@H���U�"\)B���                                    Bx^	<�  �          @�\)��
?h�����
�|  C����
@,������EC�=                                    Bx^	KD  T          @��!G�?�  ��Q��bp�C���!G�@1��]p��*�C#�                                    Bx^	Y�  �          @�\)�c33?�=q�W�� ��C�c33@>�R�=q���C                                      Bx^	h�  �          @����p��@z��E��z�C#��p��@Fff�33��Q�C�\                                    Bx^	w6  �          @��y��?����S�
�(�C��y��@<���
=���
C�                                    Bx^	��  T          @����c�
?�p��`���+�HC ���c�
@�R�1G��{C�                                    Bx^	��  �          @�(��qG�?�(��Fff���C���qG�@#33�33��ffC��                                    Bx^	�(  �          @��\�z�H?�z��9���=qC 33�z�H@�H�Q��ƸRCW
                                    Bx^	��            @�=q��33?�
=�	������C!�{��33@	����
=����C                                    Bx^	�t              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^	�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^	��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^	�f              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^	�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
	�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
X              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
&�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
5�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
DJ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
R�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
a�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
p<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
~�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
�.              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
�z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
�               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
�l              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^.�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^=P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^K�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^Z�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^iB              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^w�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�r              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
d              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^
              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^'�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^6V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^D�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^S�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^bH              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^p�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�,              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�x              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^j              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^   d          @z=q��H>�@AG�BK�C(����H�@  @=p�BE�CEO\                                    Bx^ �  �          @|���1�>�33@-p�B0=qC,�R�1녿@  @(Q�B)�RCC                                    Bx^/\  �          @��Ϳ���h�ý���33Cr�����S33��ff��33Cp!H                                    Bx^>  
(          @��
�O\)��  ?��A��RCE��O\)��=q?z�A�CJaH                                    Bx^L�  
�          @z�H�n{��Q�?��
AxQ�C5s3�n{�   ?h��AZ�\C;�{                                    Bx^[N  "          @z�H�i���u?���A��C7�i���=p�?�\)A��C?�                                    Bx^i�  
�          @~{�qG�?�?\(�AK\)C+���qG�>��?�  AmG�C1�                                     Bx^x�  T          @����X��?�
=?�Q�AɮC �
�X��>��@ ��A�\C-\                                    Bx^�@  T          @���G�?�ff?��AӅC  �G�?s33@
=B�C##�                                    Bx^��  T          @�z��c�
=u@p�A���C2���c�
�Q�@�
A��C@�                                    Bx^��  T          @�ff�j�H?���?�p�A���C#���j�H>L��@\)A���C0�3                                    Bx^�2  �          @���hQ�?��?Tz�A8  Ch��hQ�?�z�?�G�A�(�C�                                     Bx^��  �          @�33�s33?��?�ffA��
C�f�s33?(��?�p�A���C*)                                    Bx^�~  �          @����aG�?!G�?�(�A�ffC)���aG��L��@33A�C7O\                                    Bx^�$  �          @�(��B�\?��@@  B0��C)���B�\�0��@>{B.z�C@��                                    Bx^��  �          @��R�HQ�>��
@G
=B2��C.(��HQ�p��@>�RB)��CD��                                    Bx^�p  �          @���I��?J=q@L��B2��C%�3�I�����@P  B6(�C=�)                                    Bx^            @���L(�?xQ�@FffB+��C#!H�L(���{@N�RB4�RC:
                                    Bx^�  c          @���G
=?���@C�
B(�C��G
=    @U�B<
=C3�3                                    Bx^(b  
�          @�{�8��?�{@<(�B&��C���8��>��R@UBC�
C-�)                                    Bx^7  �          @���>{?��H@G�B1  C޸�>{��@UBA�C6�                                     Bx^E�  �          @��R�5�?��@Q�B=G�Ch��5����
@[�BI33C:�                                    Bx^TT  "          @�=q�AG�?c�
@Q�B8�C#�\�AG���@W�B?�C=!H                                    Bx^b�  �          @����^�R?k�@J=qB%33C%.�^�R���@QG�B,  C:�3                                    Bx^q�  �          @����`��?��\@G
=B!��C#޸�`�׾���@P��B+
=C8�f                                    Bx^�F  T          @�  �Y��?���@G�B$z�C"#��Y���k�@S33B0=qC7�H                                    Bx^��  "          @�\)�E?��@S33B1��C���E���
@c�
BD  C5��                                    Bx^��  T          @�\)�Fff?��@Q�B0(�C�)�Fff��@c�
BD
=C4��                                    Bx^�8  T          @���.�R@�@0��B33Cs3�.�R?E�@UBFz�C$\)                                    Bx^��  "          @��:=q?�ff@W�B:z�Cٚ�:=q�\)@g
=BL�\C6�                                    Bx^Ʉ  �          @�=q�7�?0��@q�BPQ�C&xR�7��^�R@p  BMz�CD�
                                    Bx^�*  T          @�G��Fff?Y��@^�RB<�C$���Fff���@a�B@�RC>�f                                    Bx^��  �          @����U?(��@W
=B2\)C(���U�:�H@UB1Q�C@Q�                                    Bx^�v  T          @��\�AG�?�{@aG�B;  C�
�AG����@qG�BM  C6�H                                    Bx^  "          @��H�<��?�(�@hQ�BB�Cz��<�;��
@s�
BP��C:#�                                    Bx^�  �          @�z��N�R>��@S�
B6�\C1T{�N�R��@FffB(
=CG��                                    Bx^!h  �          @��H�*=q?.{@g�BSz�C%���*=q�Q�@eBQ{CE�                                    Bx^0  �          @��H�C�
>.{@Y��B@�C0���C�
��Q�@L(�B0��CIW
                                    Bx^>�  T          @�=q�Fff=#�
@W
=B=�C3Y��Fff��ff@FffB*��CJ�q                                    Bx^MZ  �          @���HQ�>W
=@Tz�B:�C0{�HQ쿏\)@HQ�B-=qCG��                                    Bx^\   
�          @���Tz�>�=q@@  B(G�C/L��Tz�p��@7�B�CC�
                                    Bx^j�  �          @����a�=���@2�\B�C2c��a녿�G�@&ffB{CD�                                    Bx^yL  T          @���S�
=��
@I��B.ffC2���S�
��
=@:�HB{CG��                                    Bx^��  
�          @�z��:�H>�G�@c�
BI33C+ff�:�H��G�@\(�B@\)CF�q                                    Bx^��  �          @�����׿8Q�@
=A���C>.���׿�  ?�33A��\CHu�                                    Bx^�>  "          @��
�q녾���@(Q�B
�
C9�q녿���@�A���CG�)                                    Bx^��  �          @����}p�<#�
@�A�  C3�f�}p��u@�RA噚CA��                                    Bx^  "          @�{���Ϳ�=q?+�ACAǮ���Ϳ�  >8Q�@33CC�                                    Bx^�0  �          @�p���=q��=q?��A�p�CF!H��=q��
=?�\)A^�\CMff                                    Bx^��  
�          @�{��녿Ǯ?�
=A�G�CH������ff?n{A7�COO\                                    Bx^�|  T          @�����H���?�G�A��
CF�R���H��(�?���ATz�CM�q                                    Bx^�"  
Y          @�
=�����
=?�\)A]G�CO�
�������>B�\@�CR�=                                    Bx^�  �          @�
=�Z=q�G
=�n{�6=qC^W
�Z=q��R����Q�CX{                                    Bx^n  �          @��H�\���L�Ϳ���K�C^���\���!G��\)�݅CX{                                    Bx^)  �          @�G��dz��Fff�5�	G�C\�3�dz��#�
�����p�CW��                                    Bx^7�  
�          @�
=�j�H�;��8Q��(�CZ�f�j�H�'���=q��
=CW�\                                    Bx^F`  "          @��
�x���p�?�p�A�ffCTE�x���5>\@�=qCX#�                                    Bx^U  �          @��׿��ͽ�\)@�
=B�ffC6�H���Ϳ�Q�@��Bn�
CfW
                                    Bx^c�  
�          @���>�p�?���@��B��qB��>�p���\@�(�B��C�
=                                    Bx^rR  �          @��?�ff@
�H@���Bh�
BI(�?�ff>B�\@���B�k�@Å                                    Bx^��  
�          @�33?��?�
=@�B��RB;=q?���#�
@��HB��C�9�                                    Bx^��  T          @��R?�  @33@��HBlffBc�\?�  >���@��
B�A4(�                                    Bx^�D  
�          @�{?��?�33@�\)B�\)BP�?����ff@�\)B�ǮC�aH                                    Bx^��  �          @�
=?�33?�
=@��
BB��?�33�(�@���B��RC��                                    Bx^��  �          @�\)?�{?˅@�z�B}��B"G�?�{���@��
B��C��                                    Bx^�6  T          @�?��?&ff@�G�B��qA�G�?����33@�z�B�=qC��
                                    Bx^��  "          @�=q?Ǯ>�=q@�=qB�Q�A�H?Ǯ��  @�Q�B33C��
                                    Bx^�  
�          @�=q?��>��@��B���A��H?������@��HB��C�+�                                    Bx^�(  
�          @��H?�  >���@�33B�(�Anff?�  ��33@�33B���C�'�                                    Bx^�  �          @��H?Y��>Ǯ@�\)B���A�ff?Y���ٙ�@��RB�C�O\                                    Bx^t  �          @��׽�Q�?+�@��B�.B��ὸQ쿵@�  B�(�C�>�                                    Bx^"  
Z          @��aG�>�Q�@���B��{B�p��aG��ٙ�@�(�B�L�C�O\                                    