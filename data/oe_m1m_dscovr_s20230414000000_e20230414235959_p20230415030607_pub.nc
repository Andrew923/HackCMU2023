CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230414000000_e20230414235959_p20230415030607_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-15T03:06:07.313Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-14T00:00:00.000Z   time_coverage_end         2023-04-14T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxw���  �          @�ff��\)@`  @�z�BN��Bׅ��\)?&ff@���B��qC��                                    Bxw�f  "          @��R��@QG�@���BU�B��H��>\@���B�u�C$�\                                    Bxw�  T          @�p����@S33@��BU
=Bޔ{���>�(�@�Q�B��C!��                                    Bxw�(�  
�          @���  @8��@�  BKffB��H�  >L��@�33B��C.�f                                    Bxw�7X  "          @�p��@XQ�@��B7p�B�aH�?B�\@��
B�C"
=                                    Bxw�E�  T          @�
=���H@:�H@�  BU�B��
���H=���@�=qB�(�C1�                                    Bxw�T�  �          @�(���@�Q�@�z�B6��B£׿�?���@�33B�\Bݽq                                    Bxw�cJ  
�          @�  >�@���@~�RB(
=B�u�>�?�\@��B�(�B�ff                                    Bxw�q�  �          @��H?�33@�33@AG�A�p�B�W
?�33@7�@�{Bi��B�                                      Bxwр�  
�          @�Q�@"�\@�G�@��A���Bxp�@"�\@:=q@��BEG�BC�H                                    Bxwя<  �          @�(�@��@���?���A{33B�{@��@c33@y��B)z�B_                                    Bxwѝ�  
�          @�33?�@��@-p�A�Q�B��R?�@1G�@���B_z�Bk                                    BxwѬ�  "          @���?˅@��@K�BQ�B���?˅@z�@���Bsz�B^�R                                    Bxwѻ.  
Z          @��\?�Q�@��@S33B
  B���?�Q�@��@��Bv  BQ�                                    Bxw���  
�          @�33@ff@���@?\)A�ffB�\)@ff@=q@�(�Bb��BC��                                    Bxw��z  "          @��\@�
@�=q@2�\A�
=B|�R@�
@!�@�
=BX33B>Q�                                    Bxw��   
�          @��?��@�  @O\)B\)B�u�?��@��@�ffBv�HBup�                                    Bxw���  "          @��\?J=q@���@Z�HB�B�Ǯ?J=q@�\@���B��B���                                    Bxw�l  T          @�(�?+�@��@`��B{B��R?+�@\)@�z�B�B��                                    Bxw�  T          @��
?�\)@��R@0��A�=qB���?�\)@8��@�z�Bf�B��3                                    Bxw�!�  
�          @�33?���@��@G�A�{B�{?���@`  @�z�BB��B���                                    Bxw�0^  T          @��\?��
@�p�?�\)A���B�  ?��
@aG�@�
=B;�B|                                    Bxw�?  �          @�  ���H@>{@�
=BM�
B��f���H>��R@�33B�L�C*��                                    Bxw�M�  T          @�����\@\(�@���BH��Bܞ����\?@  @�G�B�{Cff                                    Bxw�\P  T          @�
=>���@�\)@P  B�B��q>���@*=q@��B|ffB���                                    Bxw�j�  T          @��>�\)@�p�@w�B$�\B��3>�\)?�\)@�=qB�u�B��
                                    Bxw�y�  
Z          @�(�>���@��R@fffB(�B�Ǯ>���@�\@��B��B��                                    Bxw҈B  
�          @�(�?�@�(�@Y��B�HB��R?�@�\@���B�  B|{                                    BxwҖ�  
�          @�33?�G�@��R@S33B	�B��R?�G�@=q@�\)B}�\B�p�                                    Bxwҥ�  "          @�p�?�ff@�33@.�RA��B�?�ff@B�\@�p�BcQ�B��                                    BxwҴ4  T          @�z�?�Q�@��
@#�
A��
B��)?�Q�@H��@�G�B[B��=                                    Bxw���  
(          @�33?��H@�{@�A�p�B���?��H@U@��\BM��B��H                                    Bxw�р  �          @��?˅@�  ?�
=A��B�8R?˅@dz�@��B>�B��f                                    Bxw��&  �          @�33@  @�Q�?�{A�B���@  @Y��@�(�B533Bb
=                                    Bxw���  �          @�33?^�R@��R@:=qA�=qB�?^�R@5�@�Q�BmffB��q                                    Bxw��r  T          @��\?�{@���@+�A���B�G�?�{@A�@��HBaffB��)                                    Bxw�  
�          @�=q?\)@�(�@   A�33B���?\)@K�@�\)B^�B�33                                    Bxw��  �          @�33@O\)@hQ�@N{B�B@�H@O\)?��
@��BPQ�A�ff                                    Bxw�)d  
�          @���@u@\��@��A�\)B'��@u?��@s33B'��AǙ�                                    Bxw�8
  �          @��H@Q�@|��@.{A��BI
=@Q�@�
@��\B@��B �\                                    Bxw�F�  �          @��\@7
=@�Q�@+�A���B`Q�@7
=@ff@�ffBI  Bff                                    Bxw�UV  
�          @���@h��@dz�@(Q�A�(�B1�@h��?��
@��B4=qA��H                                    Bxw�c�  �          @��\@Y��@p��@9��A�ffB?p�@Y��?�@�z�BB��A�(�                                    Bxw�r�  �          @��
@QG�@s�
@A�A��BE��@QG�?��@���BJ=qA��                                    BxwӁH  �          @���@c33@w
=@(�A�\)B=Q�@c33@Q�@���B0�A�G�                                    Bxwӏ�  T          @��@mp�@e�@!G�AиRB0(�@mp�?���@~{B/G�A�Q�                                    BxwӞ�  "          @�@j=q@|(�?�A�  B<�\@j=q@%@Z=qB{B{                                    Bxwӭ:  T          @��@>�R@��@#�
A�Q�BX�@>�R@�\@���BBz�B{                                    Bxwӻ�  
�          @��@(Q�@���@-p�A�Biz�@(Q�@ff@�\)BO
=B'Q�                                    Bxw�ʆ  �          @�=q?�@��@1G�A�z�B�.?�@AG�@�p�BhffB�L�                                    Bxw��,  �          @��?�{@��?��HA�\)B��?�{@vff@�p�B8�\B���                                    Bxw���  T          @���?�@�=q?��AU��B���?�@|��@s33B$33B�\                                    Bxw��x  T          @�Q�?�Q�@���?�A>ffB�z�?�Q�@�  @hQ�B�B�8R                                    Bxw�  �          @��Ϳ��
@��@\(�B33B��ÿ��
@@���B�u�B�=                                    Bxw��  �          @��ÿУ�@u@��B7�B���У�?��H@�z�B�u�C�                                     Bxw�"j  �          @���\@j�H@�(�BB�\B�aH��\?fff@��HB�� C�                                    Bxw�1  
�          @�{�Q�@<��@�33BP33C �f�Q�>#�
@�p�B�u�C0(�                                    Bxw�?�  T          @ƸR�*=q@5@���BK�HC)�*=q=���@�=qB���C1��                                    Bxw�N\  T          @����H@O\)@���BZ�B�{���H>�{@��RB��qC&��                                    Bxw�]  T          @����@'�@��Bm{B�
=������@�33B��{C>@                                     Bxw�k�  �          @Å���R@l(�@���BHG�B����R?k�@��
B���Cn                                    Bxw�zN  �          @��H�h��@XQ�@�ffB[{B���h��?   @�\)B�W
CO\                                    BxwԈ�  �          @\�&ff@J�H@��
Bf�RB�  �&ff>u@���B��C�f                                    Bxwԗ�  �          @��H��
=@>�R@���BqQ�B��H��
=�#�
@�=qB�(�C4ٚ                                    BxwԦ@  T          @��H���@^{@�{BZffB��H���?
=@�  B�C�\                                    BxwԴ�  T          @�녾�p�@`  @��BY��B�  ��p�?!G�@�  B�#�B�                                    Bxw�Ì  �          @�33��G�@2�\@��
By
=BŽq��G��W
=@���B��HCM�=                                    Bxw��2  T          @����@X��@���B[=qB����>��H@���B��\C�                                    Bxw���  �          @�(��&ff@,��@��BPz�C��&ff�#�
@���B��{C4@                                     Bxw��~  �          @�p��=q@)��@�  BY�CJ=�=q�\)@��B�C7\)                                    Bxw��$  T          @�ff��{@��@�p�B6G�Bؽq��{?�@��B��fC�f                                    Bxw��  �          @����H@�(�@w�B�B�Q쿚�H@	��@�z�B�ǮB��f                                    Bxw�p  
�          @Ǯ�>�R@�{@G�A�{B����>�R@ff@�z�BP�C�q                                    Bxw�*  T          @�  �>�R@���@\)A�G�B�\�>�R@S�
@�p�B3=qC��                                    Bxw�8�  T          @���
=@��?�ffA��RB�B���
=@��
@��
B6(�B��                                    Bxw�Gb  �          @�녿�@�
@aG�B`\)Bͨ���>���@�ffB���C��                                    Bxw�V  �          @�=q����?B�\@�
=B���B��
�����  @��\B��\C�H                                    Bxw�d�  "          @�p��aG�@ff@�33B��RB�.�aG��c�
@�p�B�#�Ca��                                    Bxw�sT  T          @����@C�
@�\)Bh��B�����>�@��B��C-ff                                    BxwՁ�  �          @�{�}p�@p��@��BLQ�Bъ=�}p�?p��@���B�z�CxR                                    BxwՐ�  �          @�\)�O\)@p  @���BO�B̀ �O\)?fff@ÅB��\CG�                                    Bxw՟F  T          @Ǯ�.{@1�@�Q�Bz
=Bυ�.{�u@��B���CG��                                    Bxwխ�  �          @�ff��z�?�p�@�33B�z�Cn��z���@�B��Cd�                                    Bxwռ�  T          @����\)?���@��B��fCuÿ�\)�   @�B��\Ck��                                    Bxw��8  T          @��H����?J=q@�z�B��fC#׿�����\@�Q�B��3Cp�                                    Bxw���  �          @�33����?���@�33B�u�CQ쿐�׿�  @�  B�Cm.                                    Bxw��  "          @�zῙ��?�  @��
B�W
C�H������\)@��RB��Cm8R                                    Bxw��*  T          @�zῪ=q?�G�@��RB�Q�B�#׿�=q���@�=qB�p�Ca&f                                    Bxw��  �          @θR��?5@���B���Cz���ff@�\)Bz��Chk�                                    Bxw�v  
�          @��
��G�?��
@�{B���C�׿�G���  @�=qB��fC`�)                                    Bxw�#  �          @�=q��p�?��H@�z�B�
=C�쿽p����@�\)B��C^Q�                                    Bxw�1�  �          @��Ϳ�(�?�G�@�\)B�
=C𤿼(����@��HB�C^Y�                                    Bxw�@h  �          @�\)���?�  @\B�aHB���������@�p�B��\Cb��                                    Bxw�O  �          @�
=����@ ��@�
=B��B�����Ϳ�{@�ffB��C[�)                                    Bxw�]�  �          @�����?�@�{B��fB�W
������
=@�(�B�{CZ��                                    Bxw�lZ  �          @��Ϳh��@0  @���B{G�B�p��h�þ���@ȣ�B��RCF.                                    Bxw�{   �          @�\)��ff?�z�@���B�CͿ�ff�   @�=qB�=qCrG�                                    Bxw։�  T          @��H��
=?�G�@��RB���C����
=���@�ffB�#�Cc�                                    Bxw֘L  �          @�G���(�?�z�@���B��)C�Ϳ�(����@��
B��Cl�R                                    Bxw֦�  �          @�=q��z�?�=q@��B��C����z�ٙ�@���B���CfE                                    Bxwֵ�  �          @�\)��\@�H@�  BfG�C\)��\�\@���B�(�C=xR                                    Bxw��>  �          @�p��ff@<(�@��BKG�C �3�ff>�p�@��RB��)C*�q                                    Bxw���  �          @ə�?�G�>�
=@��HB�33Av�\?�G��"�\@���Bw�
C�S3                                    Bxw��  T          @�z�?�\>L��@�33B��@ə�?�\�'�@�\)Bkz�C�
=                                    Bxw��0  T          @��?E�?��@���B��
B���?E���Q�@�ffB�B�C�k�                                    Bxw���  �          @�33?�  ?=p�@�ffB�aHBp�?�  ��
@���B�(�C���                                    Bxw�|  "          @��?���@\)@���B�W
B\)?��ͿG�@�z�B�B�C�J=                                    Bxw�"  �          @ə�>aG�@0��@�(�BffB��q>aG���  @�Q�B��
C��{                                    Bxw�*�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxw�9n              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxw�H              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxw�V�  
P          @�{���@:=q@�33BoB�녿��<�@��HB�L�C2k�                                    Bxw�e`  �          @�z�n{@J�H@��Bez�BԀ �n{>�33@���B�k�C&f                                    Bxw�t  �          @�\)�J=q@���@��\B&��B�Ǯ�J=q@33@�\)B�B�                                      Bxwׂ�  �          @�\)��Q�@�z�@��\B0=qB�W
��Q�?�{@���B�
=Ch�                                    BxwבR  "          @�Q쿚�H@�ff@�ffB6{B�(����H?���@���B�p�B��                                    Bxwן�  �          @Ǯ���\@��R@��B4Q�B�p����\?��@��
B�8RB�u�                                    Bxw׮�  "          @�
=��=q@�{@p  Bz�B����=q@Q�@�Q�Bz�B��                                    Bxw׽D  �          @Ǯ��
=@��H@hQ�BffB�\��
=@@��
Br
=Cs3                                    Bxw���  �          @�\)�	��@�p�@s�
B  B�Ǯ�	��@
=@�{Bt
=Cn                                    Bxw�ڐ  �          @�Q��33@�=q@p  B\)B�B��33@G�@��RBr��C                                      Bxw��6  T          @�G��ٙ�@��@hQ�B��B��H�ٙ�@"�\@��RBs
=B�p�                                    Bxw���  "          @�Q�^�R@�@W
=BffB�{�^�R@@  @���Bo�B�\)                                    Bxw��  �          @�녿�G�@��@K�A��
Bγ3��G�@L��@�G�Bc�B��                                    Bxw�(  
�          @ə���R@�\)@@��A�
=B��
��R@\(�@�\)Ba
=B�W
                                    Bxw�#�  �          @�ff�u@��\@*=qA�Q�B�#׽u@k�@��RBU��B��q                                    Bxw�2t  	�          @���z�@��@��A�
=B��f��z�@x��@�Q�BJ��B��{                                    Bxw�A  
�          @�z�:�H@�@Mp�A�=qB�{�:�H@E�@�  Bk�\Bή                                    Bxw�O�  �          @�녿!G�@��@N�RB33BÅ�!G�@$z�@�Q�Bx�B�p�                                    Bxw�^f  �          @�G����
@=p�@��
Bc�B�����
>k�@��B��C+T{                                    Bxw�m  "          @�Q��A�>�  @��Br�RC/5��A���\@��BQ{CY)                                    Bxw�{�  "          @�
=�Vff?n{@��B]�C$z��Vff���
@�p�BT�\CLxR                                    Bxw؊X  �          @��!G�?��H@�Q�B|ffCn�!G����R@�{Bv�HCR��                                    Bxwؘ�  �          @�{�)��?��@���Bt{CB��)�����\@�BvffCM��                                    Bxwا�  �          @ƸR�%?���@��By�C\�%��(�@�p�Bt��CQ��                                    BxwضJ  "          @����*=q@#�
@���BVp�C)�*=q�L��@�(�B�\)C5                                    Bxw���  �          @��
�*�H?�G�@��\BtC� �*�H���R@�z�By�
CL޸                                    Bxw�Ӗ  T          @�z��
=?�z�@�G�B�G�C޸�
=����@���B�B�CQ�                                    Bxw��<  �          @�33��?}p�@��\B��
Cp��녿�@�(�BzCZn                                    Bxw���  �          @�=q�8Q�>\)@��
B{33C1L��8Q��{@���BTG�C\�=                                    Bxw���  T          @˅�=p�=��
@��
Bx��C2h��=p�� ��@���BQ\)C\\)                                    Bxw�.  T          @�p��!G�?.{@��B�C$���!G��@�Q�Bm=qC[��                                    Bxw��  �          @���=p����@�Q�Bup�CG�R�=p��e@�(�B3p�Cfu�                                    Bxw�+z  �          @��
�8�ÿfff@�z�B{{CE@ �8���`  @�=qB:�HCfh�                                    Bxw�:   �          @�z��1녾�
=@�ffB�\)C<���1��E@�33BK
=Cd
=                                    Bxw�H�  �          @��
�%�\)@\B��RC7&f�%�:=q@��HBWz�CdE                                    Bxw�Wl  �          @��H�p��\)@��
B�#�C7T{�p��:�H@��
BZffCe�                                    Bxw�f  T          @ҏ\��;\)@�33B�#�C7=q����:=q@��BZ��Ce޸                                    Bxw�t�  T          @Ӆ���z�@�p�B�B�C;���Dz�@��BX�ChǮ                                    Bxwك^  
�          @���ff�\)@�B�B�CAz��ff�Tz�@�  BP�
Cj��                                    Bxwْ  �          @�\)�'
=��@�p�B�G�C@T{�'
=�Tz�@��BL��Cgٚ                                    Bxw٠�  T          @���� �׿�R@�  B�{CA��� ���Y��@�G�BMz�Ci��                                    BxwٯP  T          @�  �)���0��@���B��CB���)���Z=q@��BH\)Ch
                                    Bxwٽ�  "          @�G��4z�L��@��
B���CC�
�4z��_\)@��HBBz�Cg�                                    Bxw�̜  �          @�(��,(��E�@���B�  CC�f�,(��a�@�  BG=qCh�                                    Bxw��B  !          @׮�ff?z�@��HB�#�C$n�ff�ff@�p�Bx  CdE                                    Bxw���  �          @�=q�G
=����@��\Bh�
CH
=�G
=�Z�H@��B*Q�Cc�                                    Bxw���  
�          @ҏ\��33��H@�  B!�
CQ&f��33���\@,(�A\C_0�                                    Bxw�4  T          @�ff�H�ÿ
=q@�(�Bv�C=��H���HQ�@�  BA�RC`�                                    Bxw��  �          @�ff�<�Ϳ�@��RB}G�C>��<���I��@��HBF�RCb��                                    Bxw�$�  T          @�
=�Vff�Tz�@�Q�Bl(�CA��Vff�U�@���B5  C`�
                                    Bxw�3&  �          @�
=�.�R�^�R@�=qB�ffCE���.�R�`  @���BB33Ch�                                    Bxw�A�  �          @�{����@��
B��CQ(���z=q@�(�B;��Co)                                    Bxw�Pr  "          @ָR�Q쿗
=@��B�W
CNc��Q��s�
@�
=B?�\Cn�                                    Bxw�_  �          @У��=q��@��
B�
=CR���=q�y��@��
B4�CnO\                                    Bxw�m�  T          @����!G���=q@��B�{CO޸�!G��w�@�Q�B7p�Cl�                                    Bxw�|d  
�          @�=q�5����@�BpffCR
=�5���  @�33B&z�Cj�                                     Bxwڋ
  "          @�ff�1G�� ��@��BZ  C^E�1G����H@dz�B��Cn��                                    Bxwڙ�  �          @�z��4z��\)@���B^CZh��4z����@n�RBCm\                                    BxwڨV  �          @˅�,(����H@��Bo�\CTh��,(��~{@��B#�Ck��                                    Bxwڶ�  �          @�p��"�\�^�R@���B�B�CF�
�"�\�Vff@��BC(�Ch�{                                    Bxw�Ţ  �          @��
��Q쿎{@�B�CQ�ÿ�Q��g�@�=qBFG�Cq޸                                    Bxw��H  T          @��
�A�>�z�@�33Bz�C.s3�A���@��
BZ\)CY�
                                    Bxw���  
�          @��HQ�?��H@�p�Bj��C��HQ쿙��@�\)Bn��CI�                                    Bxw��  T          @��c33?˅@�z�BX�RC���c33�u@���Ba�\CC�                                    Bxw� :  �          @�p���ff?z�H@ʏ\B�{CY���ff��z�@ÅB�z�Cb�q                                    Bxw��  �          @����?���@���B��
C�������@���B�=qCY�q                                    Bxw��  
�          @�{�"�\?k�@�=qB���C ��"�\����@��Bw33CW�q                                    Bxw�,,  "          @�{��
?��R@ÅB���C�q��
��=q@�G�B�L�CV\)                                    Bxw�:�  
�          @�  �8Q�?ٙ�@��HBpCn�8Q쿅�@��B{CG�{                                    Bxw�Ix  T          @�����@Y��@���Bc��Bڽq����?!G�@�
=B�ffCW
                                    Bxw�X  
�          @�p�����@1G�@�  BpG�B�� ���ü�@�z�B�=qC5�                                    Bxw�f�  �          @�Q�ٙ�@�@��B}=qB�8R�ٙ����@�Q�B�ǮCA��                                    Bxw�uj  T          @�33��(�@G
=@�BX�B����(�?#�
@�G�B��fC�\                                    Bxwۄ  T          @�  �Tz�@QG�@�p�BXz�BЙ��Tz�?fff@�(�B�W
C�                                    Bxwے�  T          @�p��Ǯ@�p�@��\B$�B�uþǮ@�H@��RB�u�B��                                    Bxwۡ\  �          @�33?&ff@�G�@\)A�
=B�#�?&ff@�(�@���BB��B�                                    Bxw۰  
�          @���Mp�@G�@���BN�C�=�Mp����@�Bhz�C8�\                                    Bxw۾�  T          @�G��>{@+�@���BH�C	��>{>���@��Bu�\C-�)                                    Bxw��N  �          @�=q�QG�@>{@���B6�C	�QG�?5@��Bh�C'�3                                    Bxw���  "          @�=q�Fff@\)@i��BG�B����Fff@G�@���BVp�C�                                    Bxw��  
Z          @���\@��\?��HA�(�B�����\@�@�B%{B�aH                                    Bxw��@  "          @�=q��z�@�@&ffA�ffB���z�@qG�@�{B?�
B�3                                    Bxw��  
(          @�G��Tz�@�(���  �'�B�W
�Tz�@��?�z�A��RCJ=                                    Bxw��  �          @�Q쾮{@�녾�G�����B�����{@�@
=A���B�{                                    Bxw�%2  
�          @Ϯ�{@�ff?�(�A�
=Bފ=�{@���@�\)B$�
B��                                    Bxw�3�  
�          @�G�?�
=@�z�?���A��B�8R?�
=@�Q�@���B$p�B��                                    Bxw�B~  T          @�z�?�\)@�33@�A��B���?�\)@�(�@�p�B/�
B�=q                                    Bxw�Q$  �          @�=q?��H@���?�ffA�
B��?��H@�G�@�Q�B#(�B��\                                    Bxw�_�  "          @�?��R@�{@�\A��HB�ff?��R@�  @�z�B-�B�#�                                    Bxw�np  "          @��>8Q�@�  ?��A��B���>8Q�@�33@��
B((�B���                                    Bxw�}  "          @�(���{@�z�@�HA�  B�aH��{@���@��B:�HB���                                    Bxw܋�  �          @��
���
@�{@9��AУ�B�zὣ�
@�(�@��BM=qB��                                    Bxwܚb  
�          @��>W
=@��R@z�A�{B�\>W
=@��@�z�B9B���                                    Bxwܩ  �          @��H��Q�@�\)@Mp�A�(�B���Q�@tz�@��HBY�RB���                                    Bxwܷ�  �          @У׿�\)@�Q�@Y��B�RB҅��\)@Fff@��RBc�HB�                                    Bxw��T  "          @�=q�R�\@�{@J�HA�B�{�R�\@:=q@��B?�C
�\                                    Bxw���  	�          @�G��A�@���@L(�A��B����A�@@  @�p�BDffCY�                                    Bxw��  T          @�=q�C33@��\@N�RA�B����C33@@��@�
=BDCc�                                    Bxw��F  �          @�=q�8��@�ff@@��A�
=B���8��@?\)@��RBBffC�q                                    Bxw� �  "          @�G��*�H@��?�\)A�B����*�H@`��@l��BB��\                                    Bxw��  T          @�G��6ff@�p�@?\)A��HB���6ff@0  @�=qBD\)C�                                    Bxw�8  �          @�z��8��@��
?��HA��\B�ff�8��@e@tz�B��C �\                                    Bxw�,�  "          @��
�HQ�@�  ?c�
A��B�#��HQ�@�p�@:=qA�
=B��f                                    Bxw�;�  "          @��H��=q@�Q�?z�HA�HC�f��=q@]p�@+�A���C�)                                    Bxw�J*  "          @�G��e�@�=u?(�B��q�e�@�\)@G�A�(�C:�                                    Bxw�X�  
(          @У����@�  �!G����B�k����@�
=?�
=A��\B�aH                                    Bxw�gv  �          @���7�@W
=@�Q�B/�HC���7�?��\@���Bl�HC�                                    Bxw�v  T          @��C33@��@*�HA�B�G��C33@S33@�ffB2�HC�q                                    Bxw݄�  "          @љ��!�@�
=>���@:=qB��!�@�G�@1�A��B�33                                    Bxwݓh  
�          @�
=�:=q@��@"�\A��B�Ǯ�:=q@h��@�ffB.�
C �)                                    Bxwݢ  �          @�{�R�\@��?�p�A���B��R�\@o\)@g
=B��CT{                                    Bxwݰ�  
�          @����j�H@�=q�Y����RB�=q�j�H@�{?�
=AK\)B��\                                    BxwݿZ  �          @��o\)@��R�{��z�C p��o\)@�ff���
�0��B�W
                                    Bxw��   "          @�ff�`��@��R�%��ffB�� �`��@������l��B��=                                    Bxw�ܦ  
�          @�G��<(�@�Q��Mp��p�B�u��<(�@�\)���\�FffB��                                    Bxw��L  
�          @ə��:=q@����z��(��B���:=q@��
@   A�{B�G�                                    Bxw���  {          @�G��   @�  @
=A���B����   @�G�@�p�B+�B�aH                                    Bxw��  
�          @��H��
=@��R@|(�Bp�B����
=@:=q@�z�Bpp�B�W
                                    Bxw�>  �          @ҏ\����@��@�(�BB�Ǯ����@,(�@��Bu�\B�k�                                    Bxw�%�  T          @��
��
=@���@�RA��B�\��
=@���@���B:��B�.                                    Bxw�4�  "          @�Q��
=@�p�?\(�@�\)B���
=@��\@G
=A�Q�B�
=                                    Bxw�C0  �          @�{��p�@��H>W
=?�33B���p�@�
=@+�A�B�Ǯ                                    Bxw�Q�  �          @���p�@�  �   ��ffB�33��p�@�?��RA���B�u�                                    Bxw�`|  
�          @�G���Q�@�  �����V�\B�8R��Q�@�=q?Tz�AQ�Bڣ�                                    Bxw�o"  "          @��ÿ�@��@�
=Bw��B�#׿�=�\)@��B�=qC1�                                    Bxw�}�  
�          @θR���@�Q�@W
=B \)B�p����@?\)@�  BQ33CQ�                                    Bxwތn  �          @ҏ\�1G�@��@:�HA�Q�B�k��1G�@a�@�Q�B:�
C {                                    Bxwޛ  
(          @��H�333@��\@C33A��
B��q�333@Z=q@�33B>��CaH                                    Bxwީ�  T          @��Fff@�G�@�G�B Q�B����Fff?�(�@��Bb�C��                                    Bxw޸`  T          @��
�=p�@Mp�@�\)BB�C�f�=p�?n{@�33Bx=qC"�                                     Bxw��  "          @�Q��Tz�?�z�@�(�BXC �q�Tz�fff@�B\\)CC+�                                    Bxw�լ  �          @��
��G���@^{B	�C:���G����@>{A��HCH��                                    Bxw��R  �          @�G��S33@ ��@���BG{C� �S33>���@�  Bk�\C.B�                                    Bxw���  
�          @�33�p�?h��@��\B��C�R�p���G�@��RB|�CS��                                    Bxw��  T          @��L��?=p�@��Br33C&��L�Ϳ��@��
Be��CO8R                                    Bxw�D  �          @���ٙ�@5�@�  Bpz�B��Ϳٙ�>���@���B��HC)��                                    Bxw��  "          @ҏ\����@P  @���Be��B�{����?G�@��
B�C��                                    Bxw�-�  �          @Ӆ��  @:�H@���Bt��B�33��  >\@θRB�(�C"�H                                    Bxw�<6  "          @ָR��=q@(��@���Bt(�B��{��=q=�G�@��
B��)C0z�                                    Bxw�J�  "          @���E�@1�@���BLz�C	�3�E�?�\@��\Bw=qC*�H                                    Bxw�Y�  
�          @�p��1�@  @��HBe\)C��1녾��@���B�L�C6�q                                    Bxw�h(  "          @�\)�AG�@Tz�@�  B@G�CE�AG�?���@��Bv
=C p�                                    Bxw�v�  
�          @׮�=p�@s�
@�{B0�B��q�=p�?��@�=qBo(�C��                                    Bxw߅t  �          @�33�mp�?�(�@��BG�RC
=�mp��#�
@��
B]=qC6u�                                    Bxwߔ  �          @��
�A�@Tz�@�G�B;ffCc��A�?�z�@��RBq��C�q                                    Bxwߢ�  "          @�z��+�@h��@�=qB;G�B���+�?��H@��
Bz{C}q                                    Bxw߱f  -          @�Q����@l(�@�G�BC��B������?�z�@\B��HCk�                                    Bxw��  	�          @ڏ\����@�{@�p�B<�\B��H����?�
=@�p�B��=CQ�                                    Bxw�β  	`          @أ��@�ff@���B$�B�33�@�H@�Brz�C��                                    Bxw��X  "          @�=q�AG�@��@�
=B$33B���AG�@�@��Be��Ck�                                    Bxw���  
�          @�(��(Q�@�G�@\(�A�B��
�(Q�@a�@�\)BG�B�W
                                    Bxw���  �          @ٙ�����@�G�@���B&�B�����@{@�=qBv\)C Q�                                    Bxw�	J  "          @׮��@�p�@���B��B�z���@   @�ffBg�\C��                                    Bxw��  	�          @�=q�,(�@�ff@�G�B�
B�{�,(�@1G�@�G�B\�C5�                                    Bxw�&�  
�          @�G��,��@��@�B/(�B�Q��,��?�
=@�z�Br(�Cp�                                    Bxw�5<  T          @�  �i��?�{@���BQ{C\�i����{@�33Bc�C9@                                     Bxw�C�  T          @���J=q?�
=@��Bl�C��J=q�k�@��Bsp�CDB�                                    Bxw�R�  "          @ڏ\�7�@u@�G�B3�B����7�?�Q�@���Br{Cs3                                    Bxw�a.  �          @����mp�@P  @�G�B0�
C
�\�mp�?�33@�p�B^p�C"                                    Bxw�o�  T          @�ff�W
=@�G�@�  B��C �W
=@�@�=qBV��C0�                                    Bxw�~z  �          @޸R�-p�@��
@s�
BG�B�q�-p�@QG�@�\)BP��C��                                    Bxw��   
�          @����Vff@�ff@/\)A�p�B��\�Vff@p  @���B'�C��                                    Bxw���  
�          @��%�@�녾�ff�x��B�(��%�@���?�33A��B�33                                    Bxw�l  
�          @�ff�2�\@\�����#33B�G��2�\@�\)@�\A�(�B��)                                    Bxw�  
�          @�Q��:�H@�
=?uA�B���:�H@�@C�
A��B�\)                                    Bxw�Ǹ  �          @Å�I��@��H@�\A��B�33�I��@l��@n{B��Cs3                                    Bxw��^  T          @��/\)@`��@VffB�HB����/\)?�p�@�{BR�HC!H                                    Bxw��  �          @�G��?\)@K�@U�B�C8R�?\)?�Q�@�G�BM(�C�
                                    Bxw��  "          @�  �C33@Q�@g
=B�\C�f�C33?�@��HBSQ�CG�                                    Bxw�P  
(          @Å�8��@^{@���B'�C�\�8��?�
=@��Bb33C��                                    Bxw��  
�          @�=q�h��@J=q@�B*ffC��h��?�  @���BW�HC!�                                    Bxw��  T          @���\(�@.�R@��
B?�HC���\(�?333@�G�Bf�C(�                                    Bxw�.B  �          @Ϯ�R�\@'�@��RBF��Cn�R�\?�@��\BlQ�C*(�                                    Bxw�<�  "          @�{�=p�?��@��\Brz�C B��=p���{@�=qBq��CH�\                                    Bxw�K�  
�          @˅��33@G�@�RB�RB�\)��33@33@O\)BX
=B�L�                                    Bxw�Z4  T          @�ff>B�\@�녾��
�7�B��{>B�\@�
=@�
A�z�B�aH                                    Bxw�h�  �          @�  ��(�@��H@?\)A�B��
��(�@g
=@�p�BK(�B�33                                    Bxw�w�  
�          @�녿^�R@�?���A���B�
=�^�R@�Q�@��
B�HBǸR                                    Bxw�&  �          @У׿8Q�@�(�?���A\)B��f�8Q�@�Q�@X��A���B��                                    Bxw��  �          @�ff���\@���@i��B�B�33���\@E�@��Bg\)B�                                    Bxw�r  �          @�33��ff@���?�A��B�
=��ff@aG�@X��B*B��                                    Bxw�  �          @��׿�{@�(�>�@���B�ff��{@��\@�HA��B�                                      Bxw���  
�          @�����\)@dz�?�=qA�z�B�Ǯ��\)@(��@AG�B9B��)                                    Bxw��d  
�          @W
=��Q�@ff@��B,B����Q�?�\)@8Q�Bq��C�{                                    Bxw��
  "          @�G���{@@��@%Bp�B�Ǯ��{?���@aG�Bcz�B���                                    Bxw��  �          @��\����@\��@:�HB�B�aH����@��@~�RBk�B�z�                                    Bxw��V  T          @Ǯ�0��@�33@��B4{B�  �0��@�@�{B��BӞ�                                    Bxw�	�  
�          @ə��5@�=q@mp�B�B��5@W
=@��Bd�B��)                                    Bxw��  �          @ʏ\��@���@�=qB z�B�LͿ�@?\)@�G�BuffBǮ                                    Bxw�'H  �          @�논�@�{@i��B=qB��{��@o\)@�(�B\�RB��)                                    Bxw�5�  
�          @��>��@�G�@P��A���B�
=>��@��R@�z�BJ��B��                                    Bxw�D�  
�          @�(�?&ff@ȣ�@0��A��B��?&ff@��H@�33B3(�B���                                    Bxw�S:  �          @޸R>��
@��@(Q�A�Q�B�Q�>��
@���@���B.
=B�                                    Bxw�a�  �          @�{>��H@�
=@�
A���B���>��H@��R@��B%�B��3                                    Bxw�p�  "          @�  �0��@��
>��@�B�.�0��@�z�@��A�G�B�aH                                    Bxw�,  �          @�����Q�@z��z���Q�C&f��Q�@:=q����$��C�                                    Bxw��  
�          @���\)@�33�����L  C��\)@���>�p�@Y��C��                                    Bxw�x  �          @ȣ��Q�@��Ϳ���!�B�q�Q�@��?�G�A�Bី                                    Bxw�  T          @��
��@����#33����B�(���@��
�+���33Bѣ�                                    Bxw��  �          @�=q�33@��
�Q��{B���33@����z��Y�B�                                      Bxw��j  �          @�(��.�R@���8�����B�G��.�R@����^�R���B�p�                                    Bxw��  -          @�ff��ff@�p��1G���  B����ff@��H�W
=�33B�
=                                    Bxw��  
�          @����{@��;B�\��33C.��{@�z�?���A[33C��                                    Bxw��\  �          @ٙ���z�@��
�����\C����z�@�
=?��
A/33C�                                    Bxw�  �          @�G�����@�
=��  �
=CO\����@�Q�?=p�@���C�                                    Bxw��  �          @��H��ff@��R���H�E��C
���ff@�{>W
=?��C�{                                    Bxw� N  �          @ۅ���H@��R���yp�C	W
���H@�=q�\)��Q�C)                                    Bxw�.�  
�          @�z�����@ff�
=q��ffC�����@=p���Q��&{C�                                    Bxw�=�  �          @��H���@ ���5�ˮC J=���@8Q������\)C�{                                    Bxw�L@  �          @�(����\@z��L(���=qCz����\@R�\�
�H���CG�                                    Bxw�Z�  
�          @�=q��p�@����2�\��Q�B����p�@����O\)��
=BЮ                                    Bxw�i�  "          @�Q�5@�z��l���
=B�W
�5@�z�����B�L�                                    Bxw�x2  "          @�z��E@y���c�
�=qC s3�E@�z��(���z�B���                                    Bxw��  T          @�  ����@c33�J=q��C
������@�p���(���z�CW
                                    Bxw�~  T          @ȣ��mp�@|���G���33C33�mp�@�G���ff�e�B��                                    Bxw�$  �          @�  �0��@�=q�O\)���B�33�0��@�������V�\B�                                      Bxw��  T          @ƸR�   @�33�Tz��ffB���   @�ff��Q��UBڽq                                    Bxw��p  �          @�z��z�@�\)�Dz���B�#��z�@�  ���R�;�B��                                    Bxw��  T          @����X��@����*=q��B��q�X��@���p���	�B���                                    Bxw�޼  
�          @ƸR�hQ�@����'
=���C#��hQ�@�{�xQ���B���                                    Bxw��b  T          @�\)�@����6ff�ܣ�B��H�@�
=�p���z�B�
=                                    Bxw��  "          @�G��Z�H@�33�p���B�Q��Z�H@���:�H��
=B�{                                    Bxw�
�  T          @ə��p  @�33��33���C+��p  @�
=�W
=��33B�                                    Bxw�T  T          @ƸR�K�@�\)�������B��f�K�@��׿#�
���B�Q�                                    Bxw�'�  
�          @�=q�^{@�  ������RB�L��^{@���#�
���HB�W
                                    Bxw�6�  
�          @���E@�33�"�\����B�
=�E@�p��=p���
=B�\)                                    Bxw�EF  �          @�{�@��@���Q���Q�B���@��@��H�����FffB��H                                    Bxw�S�  
�          @��H�7
=@�  ��(���G�B�7
=@�G�=��
?:�HB��)                                    Bxw�b�  T          @�녿�\)@�{�G����HB֮��\)@�{��
=�-G�B҅                                    Bxw�q8  
�          @�����@�Q��   ��p�B�\)����@��H=�\)?�RB�k�                                    Bxw��  "          @˅��(�@�\)�޸R��  B��)��(�@�\)>���@.{B�aH                                    Bxw䎄  
�          @���z�@����+����
B޳3�z�@�z�=p��׮Bڽq                                    Bxw�*  �          @�����\@�
=�'���G�B�����\@��ÿ!G����
B�k�                                    Bxw��  �          @ʏ\�s33@���%���B�{�s33@�ff�����
B�p�                                    Bxw�v  �          @��Ϳ���@���#33��B�Q쿨��@�{��\���B�#�                                    Bxw��  T          @�(����@�z��{��=qB��f���@�z��G���=qB�#�                                    Bxw���  �          @�  �P  @��R����9B�k��P  @��H?
=@��B�L�                                    Bxw��h  �          @Ϯ��@�  ������RB�Ǯ��@���>aG�?�p�B���                                    Bxw��  
�          @��H��ff@�=q��G���{B���ff@�=q>�\)@"�\BД{                                    Bxw��  �          @Ϯ�333@�{�+�����B��333@��H?�  A5G�B�Q�                                    Bxw�Z  �          @�
=�e�@��H����z�B��3�e�@��R?��
A6ffB�                                      Bxw�!   T          @�{��z�@�Q�>�Q�@Y��C����z�@�(�?�=qA��C{                                    Bxw�/�  "          @ʏ\��
=@R�\?�ffA>�\C�q��
=@+�@�
A�Q�C�\                                    Bxw�>L  T          @�z���Q�@�{?��@��C
����Q�@qG�?�33A��C�)                                    Bxw�L�  !          @˅���\@�33��G���G�C\���\@��
?��RAYG�Cs3                                    Bxw�[�  �          @�33��{@�(�>u@\)C&f��{@�G�?�  A�  CO\                                    Bxw�j>  �          @�����33@w����H����Cc���33@tz�?L��@�z�C                                    Bxw�x�  �          @�G���z�@|�ͿE���HB����z�@\)?\)@��B�B�                                    Bxw凊  T          @��H@Q�@9���@  �	��B%�H@Q�@n{�����33BBff                                    Bxw�0  �          @��R@(��@�=q�
=�ǮBj�@(��@���Q��\)Bv                                      Bxw��  �          @�p�?�R@�ff�����Z{B�\?�R@�33?\)@�
=B�\)                                    Bxw�|  T          @�33?�R@�33��\)�m�B�u�?�R@ə�>�G�@�  B���                                    Bxw��"  �          @ƸR���
@�녿�  �{B�Q켣�
@��?�G�A�
B�Q�                                    Bxw���  �          @\?���@�ff�G�����B�?���@��\�k���B�33                                    Bxw��n  T          @љ�@+�@����K���
=Bw��@+�@�G�����DQ�B�8R                                    Bxw��  T          @У�@U�@��H�1���z�B]�@U�@�
=�����(�Bj�
                                    Bxw���  T          @�\)@^{@��\�g��{BF��@^{@���������B]��                                    Bxw�`  
�          @���@W
=@��R�i���ffBMp�@W
=@����
=��33Bcz�                                    Bxw�  �          @�{@[�@�z��`���p�BIQ�@[�@�G��   ��33B_                                      Bxw�(�  "          @�=q@W�@��R�S33��BMp�@W�@�G����
��33Ba�                                    Bxw�7R  �          @��
@*=q@�G��*=q��p�Bs��@*=q@�z῀  �\)B~�                                    Bxw�E�  
�          @�p�@E�@��O\)����BV�@E�@�  �޸R��Q�Bi\)                                    Bxw�T�  �          @Å@K�@;������2�B*��@K�@�(��G����BQ�R                                    Bxw�cD  T          @�{@p�@\������:�\BZQ�@p�@�ff�J=q���RBy�\                                    Bxw�q�  �          @��@j=q@�  �i���  B>(�@j=q@�ff�����BV{                                   Bxw怐  T          @�(�@qG�@����h����B;��@qG�@�
=�(���ffBSG�                                   Bxw�6  �          @�p�@e@���s33�
�BD�
@e@����33����B\Q�                                    Bxw��  
�          @�{@Tz�@`�����H�.33B:z�@Tz�@����P����{B\�
                                    Bxw欂  �          @Ӆ@5@^{�����;BJ��@5@����^{��33Bm\)                                    Bxw�(  �          @��@H��@��H�u���BR�@H��@��H�����Biff                                    Bxw���  T          @��@c�
@���333��{BLp�@c�
@�녿��
�;
=B[��                                    Bxw��t  "          @��@\)@x���;���B0�H@\)@�33�����g\)BDG�                                    Bxw��  �          @ə�@w�@�33�p���Q�BA(�@w�@�=q�@  ��p�BL��                                    Bxw���  
�          @���@q�@�=q���Q�BI�\@q�@�=q�O\)����BU�                                    Bxw�f  �          @���@�  @�
=�����\)B@��@�  @��5����BK�R                                    Bxw�  T          @��H@b�\@�{�"�\��BY�@b�\@���c�
��G�Bd��                                    Bxw�!�  
�          @���@�@�
=�&ff��
=B��\@�@�Q�^�R��B�\)                                    Bxw�0X  "          @���@w�@p��?\A{�B0�@w�@G
=@&ffA�p�B\)                                    Bxw�>�  �          @�z�@h��@��H��Q��aG�BNff@h��@�\)?���A'�BK��                                    Bxw�M�  �          @�Q�@�33@Q녿s33���B��@�33@Z=q=���?�  B(�                                    Bxw�\J  T          @�G�@~�R@�\)�fff���BAff@~�R@�G�?�\@��BB��                                    Bxw�j�  �          @���@l(�@�33��=q�K�
BM  @l(�@�G�>�?��
BQp�                                    Bxw�y�  �          @�{@a�@��Ϳ��\�BS{@a�@��>�ff@�33BU(�                                    Bxw�<  
�          @�p�@a�@��
��33�2�\BRp�@a�@�  >��R@AG�BU�                                    Bxw��  "          @�  @Tz�@�녿�ff�Hz�B]�@Tz�@�\)>aG�@Q�Ba(�                                    Bxw祈  T          @�  @l(�@�z῝p��5��BS�
@l(�@���>��R@5�BW                                      Bxw�.  
�          @��H@a�@��H�L����z�BW�@a�@��?333@�(�BX
=                                    Bxw���  �          @�G�@S�
@�(������(  Bd��@S�
@��>��@��\Bf�R                                    Bxw��z  �          @�z�@Mp�@�33��\)�Ep�Bk��@Mp�@�Q�>��R@0��Bo                                      Bxw��   T          @�z�@�@�=q�����B�k�@�@��u��B��R                                    Bxw���  �          @�  ?�  @��\�5���G�B���?�  @���s33�{B��                                    Bxw��l  �          @�33@333@�Q��(���\)B�\@333@θR������B�.                                    Bxw�  �          @�p�@`  @�  ����(�Bf  @`  @�����p�Bm�                                    Bxw��  �          @�{?W
=@��
�x���	�B��?W
=@�����\����B�B�                                    Bxw�)^  �          @��L��@�������RB��L��@����G����B�k�                                    Bxw�8  �          @߮>���@�{��33�=qB��>���@Ϯ�"�\���RB��H                                    Bxw�F�  "          @ڏ\?k�@��R�����G�B�\?k�@�  �#�
��ffB�Q�                                    Bxw�UP  �          @�  ?���@�
=��(��.33B��{?���@�(��A���(�B�                                      Bxw�c�  �          @�  @ff@���~�R���B�aH@ff@��
����{B���                                    Bxw�r�  "          @�ff@HQ�@��\��\�o�Bw  @HQ�@�33<��
>L��B{(�                                    Bxw�B  
�          @�=q@s33@�z�?�z�A�Bm{@s33@���@?\)A�
=Bb�                                    Bxw��  �          @���@hQ�@��>��@�Bq�@hQ�@���@Q�A�Q�Bl33                                    Bxw螎  H          @�  @P  @�{?��@��\B�.@P  @�
=@!�A�ffBy�R                                    Bxw�4  
�          @�\@>�R@��
?Y��@�p�B��{@>�R@��H@0  A�z�B�R                                    Bxw��  �          @�33@8��@�{>.{?���B���@8��@Å@�\A�{B�\)                                    Bxw�ʀ            @���@-p�@��H�8Q쿷
=B�33@-p�@�33?�  Ac
=B��q                                    Bxw��&  �          @��
@[�@�=q����  BrG�@[�@�z�?.{@��Bs\)                                    Bxw���  
�          @�{@O\)@��H�!���{Bs��@O\)@ʏ\�:�H��(�B{ff                                    Bxw��r  T          @��@E�@�\)�a���Bm��@E�@��ÿ�\)�w\)B{�                                    Bxw�  �          @陚@O\)@����Q���Bp�\@O\)@˅��G��?
=B{�
                                    Bxw��  `          @�(�@N{@����4z���p�Bw�\@N{@ҏ\�u��Q�B                                    Bxw�"d            @���@Fff@���*�H���RBz��@Fff@�Q�Tz��ӅB�{                                    Bxw�1
  T          @޸R@*=q@���@���Σ�B�.@*=q@Ǯ���\�)�B��R                                    Bxw�?�  T          @�{@4z�@��R���R��HBlff@4z�@�
=�+����\B                                      Bxw�NV  
�          @ʏ\@�@���E���33B�  @�@��ÿ�  �Z�RB�                                    Bxw�\�  T          @Ϯ@�@�ff�s�
���B~z�@�@��H��
���B��q                                    Bxw�k�  �          @У�@z�@����ff�-Q�Bq��@z�@�ff�Fff��Q�B��)                                    Bxw�zH  
(          @�G�@P��@����/\)���HBm�R@P��@�=q����\)Bw                                      Bxw��  
�          @��H@j=q@���������Bh�R@j=q@ə����H�vffBoQ�                                    Bxw闔  
�          @�ff?p��@vff��z��`G�B�k�?p��@�������B�                                      Bxw�:  
�          @���?�
=@{����R�T��B�Ǯ?�
=@�(����
��B�ff                                    Bxw��  H          @�G�@#33@��\����0{Bn33@#33@����\(����
B��                                     Bxw�Æ  	�          @�@,��@���(��	(�B~G�@,��@��
�(���p�B��                                    Bxw��,  T          @�{@/\)@��z=q��ffB�p�@/\)@�G��
=q��\)B��=                                    Bxw���  "          @�{@=q@ҏ\�'
=��\)B��3@=q@ᙚ�(�����\B�.                                    Bxw��x  "          @�(�@�ff@�{�������\BF�\@�ff@��
��R���
BY�                                    Bxw��  
(          @�p�@xQ�@�p���z���\BT�@xQ�@��
�!���(�Bf�\                                    Bxw��  
�          @�(�@o\)@�=q��G��	
=BV�@o\)@�=q�-p����\Bi�                                    Bxw�j  �          @�=q@Z�H@�\)����\)B^{@Z�H@����@  ����Br�                                    Bxw�*  T          @�@>{@�\)��p��Q�Bl@>{@\�G
=���B�
                                    Bxw�8�  
�          @陚@1G�@��������(��Bl�@1G�@�=q�Z�H�ߙ�B�#�                                    Bxw�G\  �          @�Q�@?\)@��R���   Bfff@?\)@�=q�L����ffB{=q                                    Bxw�V  �          @�ff@4z�@�=q��  �%�Biz�@4z�@��R�S33�ۅB~�H                                    Bxw�d�  �          @�p�@2�\@�(�����"�Bk�@2�\@���L����\)B�{                                    Bxw�sN  
�          @�ff@(�@�  �����  B�H@(�@���@  ��G�B�{                                    Bxw��  �          @��@*=q@�  �{���HB�L�@*=q@��
�G����B��{                                    Bxwꐚ  
�          @�@1�@�G���G���
Bt�@1�@����0  ��(�B�p�                                    Bxw�@  T          @�z�@33@��
��Q��{B�k�@33@���<(�����B�33                                    Bxw��  �          @�Q�?��@�{��G��,z�B��?��@����P����{B�{                                    Bxw꼌  
�          @޸R@6ff@y����\)�7�\BWp�@6ff@���n�R�=qBsff                                    Bxw��2  "          @���@1�@|������9=qB[ff@1�@�p��s33�z�Bv��                                    Bxw���  �          @�?�33@�33�����N{B�8R?�33@�ff�����B�aH                                    Bxw��~  T          @��
@^�R@7�����A�B33@^�R@�(����H��BG��                                    Bxw��$  "          @ᙚ?Ǯ@�p���ff�?�
B�33?Ǯ@���tz��
=B�=q                                    Bxw��  �          @�\)?��@�  ��=q�2��B�aH?��@��a���RB�(�                                    Bxw�p  
�          @�R?�ff@�33���
�6Q�B���?�ff@����hQ���\)B��R                                    Bxw�#  �          @�
=?���@�Q����R�.�\B��?���@���[���Q�B���                                    Bxw�1�  
�          @�R@�@�����  �;��B~p�@�@�z��xQ���B��R                                    Bxw�@b  �          @���?�=q@����ff�<�B���?�=q@����s33�ffB�z�                                    Bxw�O  �          @�\@E�@��R���
=B]p�@E�@�\)�E����HBr��                                    Bxw�]�  
(          @��@_\)@���s33�
=BZ=q@_\)@�  �
=���RBj                                    Bxw�lT  �          @��?8Q�@�Q����H�)�\B�k�?8Q�@�33�QG�����B�{                                    Bxw�z�  "          @�  ��G�@�Q�����9  B�z��G�@�
=�n{���B�                                      Bxw뉠  "          @�\)>.{@�G���p��6��B�{>.{@�\)�j=q��33B��
                                    Bxw�F  "          @�33>��@�  �����4
=B���>��@����aG����B�W
                                    Bxw��  "          @�?B�\@������#\)B�aH?B�\@�{�@  �ͅB�\                                    Bxw뵒  �          @�>�33@�����  �-�B�#�>�33@�33�P����ffB��                                    Bxw��8  
�          @�G���
=@\�������b=qB��Ϳ�
=@�G����'B�                                    Bxw���  �          @ۅ����@����ff�Op�B�W
����@����|����B��R                                    Bxw��  �          @޸R��ff@�����H�3  B�uþ�ff@��R�Y�����B��=                                    Bxw��*  �          @�Q쿇�@�{���H���B��ῇ�@�=q�!���ffB���                                    Bxw���  �          @�33���
@�G��p����
B�Q���
@�z��(��`  B�
=                                    Bxw�v  �          @�33��ff@�z��L����B��)��ff@�Q쿾�R�C
=Bɽq                                    Bxw�  
(          @�\���@��H��p��3�B�Ǯ���@��R�c�
��G�Bأ�                                    Bxw�*�  �          @�{� ��@S�
�����Q��B�B�� ��@��H�����B�33                                    Bxw�9h  �          @�(��(�@G���  �fp�B�33�(�@������2p�B�G�                                    Bxw�H  T          @�=q� ��@W
=���H�`�\B�Ǯ� ��@�Q������+  B��)                                    Bxw�V�  �          @�z��#33@s33��
=�HffB��#33@��\������B�Q�                                    Bxw�eZ  T          @��1�@Z=q���R�Q��C#��1�@�Q���z�� ffB�p�                                    Bxw�t   �          @�\)�w
=?�p�����Vz�Cٚ�w
=@H����33�6�
C�\                                    Bxw삦  �          @�\�p  ?�\)���H�^��C�)�p  @4z���G��A��C�q                                    Bxw�L  �          @�\)�s33?���\)�\��C"�f�s33@&ff����B�C��                                    Bxw��  
�          @�=q�`��?��
��33�`{C{�`��@N{��ff�=��C	�                                    Bxw쮘  "          @����[�?�=q��\)�Z��C�H�[�@J�H���\�7��C	B�                                    Bxw�>  T          @Ӆ�w
=?�=q��p��N��C ���w
=@&ff��p��4Q�C                                      Bxw���  �          @����p  ?=p������X��C(�H�p  @�
��{�DffC=q                                    Bxw�ڊ  �          @�������?aG������Pp�C'�3����@(�����;��Cp�                                    Bxw��0  
(          @�(���?�p���=q�J��C#����@!����\�2�C��                                    Bxw���  `          @أ�����?�Q���z��F{C �H����@,(�����,{C�                                    Bxw�|            @�����z�?u��33�H�
C&�R��z�@p���{�4
=C޸                                    Bxw�"  
�          @���
=q@(Q������\�Cff�
=q@q����\�,�B�                                    Bxw�#�  �          @�\)�J=q@�{�����;p�B�aH�J=q@�ff�W
=�{B�L�                                    Bxw�2n  T          @�=q�p��@}p�����?=qBΨ��p��@��R�W
=�ffB�p�                                    Bxw�A  �          @��;�p�@�
=���+��B����p�@�(��;���Q�B��                                    Bxw�O�  T          @�ff����@�(���ff�/33B�����@��H�J=q��
=B�=q                                    Bxw�^`  T          @�G����H@k���Q��FG�B�Ǯ���H@�  �l���Bۊ=                                    Bxw�m  �          @�\)��(�@   ��z��/\)C�{��(�@dz��w��=qC!H                                    Bxw�{�  T          @ڏ\����@�R��G��3\)Cz�����@Vff���H��C�                                    Bxw�R  
�          @������?�p����
�H�C������@P  ���R�)Q�C
                                    Bxw��  T          @ᙚ�s33@G���\)�Oz�C��s33@S�
����.�C
��                                    Bxw���  T          @�ff�tz�@%���R�<\)C޸�tz�@n�R��p���HC��                                    Bxw��D  T          @ٙ��mp�?z�H���H�^(�C%0��mp�@����G
=C޸                                    Bxw���  T          @�33�u�?������\�L��C��u�@L����{�,�
C{                                    Bxw�Ӑ  �          @��S�
@.�R��z��N  C���S�
@~{����%��C��                                    Bxw��6  �          @�\�n{@~{�����\)C&f�n{@��R�Z�H��\)B��R                                    Bxw���  �          @��H�@��@������'\)B�ff�@��@�33�Z=q��B�
=                                    Bxw���  T          @�  �C�
@��
���
�!B�8R�C�
@���N�R��33B�L�                                    Bxw�(  T          @���.{@u��G��;�B�� �.{@��R�|�����B�k�                                    Bxw��  
�          @���333@�������B����333@�G��:=q��p�B�R                                    Bxw�+t  "          @��H�!�@������!33B����!�@�
=�Dz����HB�Ǯ                                    Bxw�:  �          @ҏ\�  @�33����&��B��  @����J�H��\)B�(�                                    Bxw�H�  T          @�ff���R@�  �����+{B�����R@�
=�U����B�{                                    Bxw�Wf  T          @��H�Tz�@���hQ�� =qB��Tz�@�\)����z�B�u�                                    Bxw�f  T          @ڏ\�hQ�@��
�\����\)C 33�hQ�@����R����B��R                                    Bxw�t�  T          @��H�W�@����=q��C z��W�@�33�;��̸RB���                                    Bxw�X  T          @�(���p�@�ff���\�;  B��)��p�@���j�H���Bؙ�                                    Bxw��  .          @��� ��@��R���R�/�B�� ��@�
=�c�
���RB�aH                                    Bxw  �          @�=q�\@����p��$�HB�.�\@�\)�HQ�����B���                                    Bxw�J  
�          @��ÿ�z�@��������\)B��쿴z�@��H�(Q�����B�                                      Bxw��  "          @�Q�\@��H������
B�(��\@�{�3�
���HBШ�                                    Bxw�̖  �          @�  �E�@���:=q��p�B�aH�E�@�G�����g�
B�#�                                    Bxw��<  T          @�ff��\@���U���
B�\��\@��R��(���p�B�                                      Bxw���  �          @����Z=q@����'�����B��R�Z=q@�����
�.{B�Q�                                    Bxw���  "          @�Q��HQ�@�G��1��\B�8R�HQ�@�����Q��EG�B�R                                    Bxw�.  
�          @أ��O\)@���4z��ď\B�=�O\)@�  ���R�K�B���                                    Bxw��  
�          @ٙ��H��@�{�A���(�B�L��H��@�Q���H�iB�(�                                    Bxw�$z  �          @�=q�u�@�\)�(����z�B�
=�u�@�
=����<  B���                                    Bxw�3   �          @ٙ���  @��R�7���33CW
��  @�Q��
=�c�B��\                                    Bxw�A�  T          @��
��z�@���U���  C�
��z�@����G���(�C�                                    Bxw�Pl  "          @�������@p���s�
��C	�\����@���4z��ď\CY�                                    Bxw�_  �          @У��\)@^�R�p  �G�C
�f�\)@�Q��4z��Σ�C
                                    Bxw�m�  �          @�G�����@fff�j�H�C
G�����@���.{���CǮ                                    Bxw�|^  
�          @��H�U@�G��\)��C�
�U@���;���\)B���                                    Bxw�  �          @�33�X��@vff�����CW
�X��@�
=�H����33B�\)                                    Bxw  "          @��
�2�\@��������z�B��2�\@�\)�:=q��\)B�33                                    Bxw�P  "          @Ӆ�^{@��l(��C���^{@��'�����B�L�                                    Bxw��  "          @�33�@��������"�B�{�@����HQ���  B��f                                    Bxw�Ŝ  
Z          @�p����R@����z��{B����R@�z��:=q�υB��
                                    Bxw��B  T          @�p�?��@�Q��tz����B�Ǯ?��@����*�H���HB��{                                    Bxw���  
Z          @�{@p��@vff�Tz���G�B6@p��@�������\)BH�                                    Bxw��  	�          @��
>�ff@�ff��Q��!G�B��>�ff@�  �3�
��B��                                    Bxw� 4  
�          @ҏ\�@�  �fff�ffB�ff�@�ff�ff���B�Q�                                    Bxw��  
�          @љ���@�Q��X����z�B���@�������33B�                                     Bxw��  
�          @У��$z�@����dz��{B�Q��$z�@��������B��                                    Bxw�,&  
�          @Ӆ�,��@����`  �B�,��@�\)�33���B�aH                                    Bxw�:�  
�          @����e@��
�>�R��p�B��
�e@����
�t(�B�
=                                    Bxw�Ir  
�          @����\��@�z��@����G�B���\��@�{��  �k33B�\)                                    Bxw�X  "          @���{@�
=��H��
=C �R��{@�z῔z��B�.                                    Bxw�f�  �          @�=q�j�H@����6ff����B�\)�j�H@�녿Ǯ�Lz�B�                                     Bxw�ud  
�          @�\�hQ�@�=q�:�H����B����hQ�@��H��\)�TQ�B��                                    Bxw��
  
�          @��7
=@�ff�c33����B���7
=@��
�G����B���                                    Bxw�  
�          @��H�>�R@���fff��{B�q�>�R@�p��
=��  B�k�                                    Bxw�V  	�          @���'
=@�ff��=q��B���'
=@�=q�H����Q�B�L�                                    Bxw��  
N          @�z���@�ff��ff�!��B�q��@��H�P����
=B�aH                                    Bxw�  !          @�33���@�Q����R�"p�B��ÿ��@�z��L����
=B΀                                     Bxw��H  "          @ۅ����@�����\)B��f����@����8Q���
=B��                                    Bxw���  �          @�����H@�Q���\)���B�����H@\�:�H��ffB�{                                    Bxw��  
�          @�(���  @����(��p�B�LͿ�  @����3�
����B�                                      Bxw��:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxw��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxw��  �          @�=q��ff@�z��������B����ff@���-p���\)B�k�                                    Bxw�%,  
�          @�Q�(�@�
=�z=q��\B����(�@�ff�%�����B�(�                                    Bxw�3�  T          @��ÿ&ff@�G����ffB�  �&ff@��H�8Q��ɅB�.                                    Bxw�Bx  �          @�  �u@���w
=�p�B�.�u@�z��#33��ffB���                                    Bxw�Q  "          @��H���R@��H�u�	�B�L;��R@���\)��=qB��{                                    Bxw�_�  T          @�녿B�\@����y����RBøR�B�\@�Q��#�
���RB��f                                    Bxw�nj  
�          @�{��@�ff�W
=��B����@���   ���B؀                                     Bxw�}  �          @�
=�p�@�
=�Tz��㙚B�8R�p�@�녿��H���\Bڊ=                                    Bxw�  
�          @�p��\)@���Z�H���B��\)@Å�
=��B�G�                                    Bxw�\  
M          @�
=�'
=@����Vff��B���'
=@�(�������B�.                                    Bxw�  �          @�
=�9��@����HQ��ՅB�u��9��@�=q��=q�r�HB�
=                                    Bxw�  �          @�ff�A�@�p��G
=���HB�ff�A�@�\)��=q�t  B�                                    Bxw��N  
Z          @�\)�8Q�@���K���G�B�ff�8Q�@�녿���z�\B��H                                    Bxw���  �          @�Q��J=q@�{�E��\)B�L��J=q@�����o
=B랸                                    Bxw��  �          @���Mp�@�33�L����33B����Mp�@���
=��{B��                                    Bxw��@  �          @�Q��Y��@�  �H�����B����Y��@�=q��33�{�B�                                    Bxw� �  �          @�
=�G�@���I���ׅB�ff�G�@�p�����|  B�=                                    Bxw��  
�          @�Q��#�
@�  �`����Q�B����#�
@�(��������B�L�                                    Bxw�2  
�          @�p����@���c�
��z�B㞸���@�=q����=qB���                                    Bxw�,�  �          @����
=@��
�hQ���{B�u��
=@�G��
=���B޳3                                    Bxw�;~  �          @ۅ��
@�ff�u���B�����
@�p��%��G�Bޙ�                                    Bxw�J$  �          @ۅ�
=@��tz��z�B�
=�
=@�z��%��\)B߳3                                    Bxw�X�  
�          @�z��
=@�ff�\)��B���
=@�ff�0  ��Q�B���                                    Bxw�gp  �          @�z��p�@�33�|(��{B��Ϳ�p�@\�*�H��
=Bӳ3                                    Bxw�v  T          @���p�@�ff�|(��z�B����p�@�{�,������B�                                    Bxw�  
�          @�{����@��H����p�B�  ����@���E��p�Bؔ{                                    Bxw�b  
�          @�
=��@��������RB����@�(��E���HB��                                    Bxw�  "          @޸R�Q�@�(����� �B�8R�Q�@�  �Tz���G�Bܽq                                    Bxw�  �          @߮���H@�=q���
�(�B�=q���H@���J=q��33BظR                                    Bxw�T  
�          @�ff���R@�����\�33B�����R@�z��G��ՙ�B�ff                                    Bxw���  �          @�(���=q@���� ��Bܨ���=q@����P  ���
B�{                                    Bxw�ܠ  
�          @�z��
=@������\�&�\B�녿�
=@���[���RBٔ{                                    Bxw��F  �          @�=q��(�@�ff�����'{B�ff��(�@��\�Z�H��=qB��
                                    Bxw���  
Z          @�  ��\@�����p��#B���\@��
�QG���ffB�                                    Bxw��  �          @��
�z�@��H���H���B�=q�z�@�p��K���z�B�\                                    Bxw�8  �          @�z��.{@�G��������B��)�.{@�z��S�
���B�{                                    Bxw�%�  �          @�z��\)@����{�-�B�(���\)@����e���=qB�u�                                    Bxw�4�  "          @�(����@�����H�3�B�����@�G��k��=qB�Q�                                    Bxw�C*  
�          @�(���p�@��������633B��)��p�@�  �p  �G�B�ff                                    Bxw�Q�  "          @ڏ\�L��@��H��G��>��B�(��L��@�=q�{���
B���                                    Bxw�`v  �          @��;�=q@�\)����E�\B��
��=q@�Q�������B��{                                    Bxw�o  �          @���Y��@�G���33�J�
B�G��Y��@�33��G���B���                                    Bxw�}�  T          @��
�k�@��\��Q��G�HB��ÿk�@�����R��B�p�                                    Bxw�h  
�          @��8Q�@�\)��{�N�B�B��8Q�@���������B�k�                                    Bxw�  �          @���ff@�G������S��B�#׿�ff@�(������#{B��                                    Bxw�  	�          @�\)��ff@���
=�B  B��쿦ff@�ff��z���B��                                    Bxw�Z  T          @�녿�=q@�Q����H�QG�B�
=��=q@��
���H�"Q�B�W
                                    Bxw��   T          @����  @r�\����YG�B���  @�ff����+33B��f                                    Bxw�զ  �          @�ff���
@p����(��Z�RB�3���
@�{���,�
B۳3                                    Bxw��L  �          @�\�޸R@z�H��{�X�RB��
�޸R@�33���R�*p�Bٞ�                                    Bxw���  T          @�\��Q�@�G����
�H{B�q��Q�@�����=q��B۞�                                    Bxw��  �          @�33��z�@��\����Az�B�녿�z�@����(��ffB���                                    Bxw�>  "          @��
�H@�=q��=q�9�HB���
�H@�33��
=�ffB�p�                                    Bxw��  
�          @�(���p�@�ff�����<��B�z��p�@�  �������BՏ\                                    Bxw�-�  �          @�Q���H@�=q����D�\B�����H@�����{��RB���                                    Bxw�<0  
�          @�׿�p�@��������FQ�B���p�@�33������Bܙ�                                    Bxw�J�  T          @�
=��=q@������H�K�B�z��=q@�  ��=q��B�\)                                    Bxw�Y|  "          @�Q����?�p����
�M{Ch����@8�����
�5(�C5�                                    Bxw�h"  	�          @�\�p  @6ff���H�G�\C�q�p  @~�R��33�&\)CJ=                                    Bxw�v�  
�          @�33�i��@J=q�����C�\C+��i��@�����\)� z�C�{                                    Bxw�n  
�          @��
���
@%��  �B{Cٚ���
@l����=q�$=qC

=                                    Bxw��  T          @�R��=q@'���33�?p�C8R��=q@l����p��!G�C	�R                                    Bxw���  �          @����z=q@\����\)�4Q�C
�=�z=q@�����
�G�C�                                    Bxw��`  �          @��
�s33@XQ���=q�9G�C
Y��s33@���\)�
=C�
                                    Bxw��  �          @�\)�]p�@n�R�����2p�C���]p�@�
=����ffB��                                     Bxw�ά  �          @�  �XQ�@�z������)��C:��XQ�@��H�z=q�Q�B�33                                    Bxw��R  T          @�
=�e@tz����R�-�HC8R�e@�����G��(�B��3                                    Bxw���  �          @�\)�^�R@~�R��p��+�C#��^�R@�{�}p��  B�\)                                    Bxw���  �          @�R�y��@j=q���\�(G�C�=�y��@�33�|(����CB�                                    Bxw�	D  �          @���mp�@n{��33�*�C��mp�@���|���33C �                                     Bxw��  �          @���`��@w
=�����,��C@ �`��@�=q�~{��RB�(�                                    Bxw�&�  T          @���XQ�@s33��G��2�HC�
�XQ�@�G����
�ffB�ff                                    Bxw�56  �          @���c33@p  ���R�/=qCp��c33@�
=�����	�RB���                                    Bxw�C�  �          @�\)�`  @y����\)�.Q�C޸�`  @��
��G��
=B�\)                                    Bxw�R�  
�          @�G��\(�@������*
=C�3�\(�@����{���B�z�                                    Bxw�a(  �          @��g�@��
��ff���C ��g�@�ff�W����
B��                                    Bxw�o�  �          @�=q�`  @�������Q�B����`  @�=q�L���ϮB�L�                                    Bxw�~t  "          @��
�R�\@�z�����B��
�R�\@��R�S33�ԸRB���                                    Bxw��  �          @���b�\@�33���H�p�B�B��b�\@���N{��B�{                                    Bxw���  
�          @���q�@�33��=q�p�B����q�@���L���ɅB�z�                                    Bxw��f  �          @�p����H@�p������\C0����H@����Y�����HB��                                    Bxw��  T          @���33@����=q�C�3��33@���dz���  CaH                                    Bxw�ǲ  T          @�  ���@��H������C�����@�{�^{��Q�C �R                                    Bxw��X  "          @�����@������ffC�����@�G��_\)��Q�B�Ǯ                                    Bxw���  T          @�Q��z=q@�������33C��z=q@�G��g���Q�B��
                                    Bxw��  "          @�ff�|(�@�ff���R�(�C��|(�@���n{���
B�G�                                    Bxw�J  �          @�
=�~�R@�(������33C��~�R@����s33����C @                                     Bxw��  T          @�ff�fff@������� {C33�fff@��\�o\)��B��                                    Bxw��  
�          @�ff�XQ�@�p���{�z�B����XQ�@���fff��B�                                    Bxw�.<  �          @�{�g
=@�ff��  ��RB�
=�g
=@����Y����B�#�                                    Bxw�<�  T          @��H�S�
@���=q�$(�B��=�S�
@�33�qG�����B�=                                    Bxw�K�  �          @��fff@�Q�������HB�B��fff@�33�[���B�p�                                    Bxw�Z.  T          @����h��@����  �  B�B��h��@����X����  B��{                                    Bxw�h�  4          @�{�g�@���������B�Q��g�@��R�N{��Q�B�                                    Bxw�wz  �          @���fff@����H�
�B����fff@�\)�I����ffB�#�                                    Bxw��   �          @�ff�b�\@�����H��\B��f�b�\@�p��Z�H��33B�                                    Bxw���  T          @��b�\@����z���B���b�\@���^�R��B�G�                                    Bxw��l  �          @�p��\��@�G���z���B����\��@���^{����B�                                    Bxw��  �          @���J=q@��
����{B���J=q@�
=�XQ��Σ�B��
                                    Bxw���  �          @�\)�:=q@�33��ff��RB���:=q@�p��J�H��B���                                    Bxw��^  �          @����@�  ���=qC+���@����k��㙚C��                                    Bxw��  �          @�(����@�������Ch����@����s33��RC��                                    Bxw��  �          @����=q@������
�	�RC����=q@��R�O\)��(�B�                                      Bxw��P  �          @�  �|��@�
=��
=��C��|��@���g���z�B�L�                                    Bxw�	�  �          @�\)�vff@��\������HC �=�vff@�ff�aG���
=B�                                    Bxw��  �          @������@�����ff�=qC^�����@���Y������C ��                                    Bxw�'B  �          @�������@������
=C�3����@�ff�h���㙚Cc�                                    Bxw�5�  �          @����
@�{��Q��  CxR���
@�33�qG���
=C ��                                    Bxw�D�  �          @��aG�@��
���� �HB��3�aG�@����u���B���                                    Bxw�S4  �          @�=q�e�@��R��
=�#��C ��e�@���z�H��
=B�
=                                    Bxw�a�  �          @�=q�[�@�������!ffB����[�@��\�tz����
B�#�                                    Bxw�p�  �          @���y��@����z���C xR�y��@�p��@�����HB�
=                                    Bxw�&  �          @�G��l��@�z������C �\�l��@����e���\B���                                    Bxw���  �          @����e�@�
=����B�\)�e�@�33�c�
��\)B�33                                    Bxw��r  �          @�\��G�@��\�����RC���G�@�33�B�\��Q�B��                                    Bxw��  T          @�G��L��@�z�����RB�p��L��@���<����(�B�                                    Bxw���  �          @�Q��`  @�G�������\B��=�`  @��\�G���
=B��                                    Bxw��d  �          @�
=�G
=@�����R�(��B�Q��G
=@�{�y����33B�u�                                    Bxw��
  �          @�{�\)@�ff��z���C���\)@����Vff����B�=q                                    Bxw��  �          @�ff�[�@�\)��=q�
=B�#��[�@����J�H��B�ff                                    Bxw��V  �          @�z���G�@�p��i�����HC^���G�@��\�!���\)B��q                                    Bxw��  �          @��
��33@�{�������CW
��33@����a���{C�                                    Bxw��  �          @�����@\(���
=�"33Cn���@���xQ��ffC��                                    Bxw� H  �          @������@aG���\)�+�HC
�=����@�����
�	ffC�\                                    Bxw�.�  �          @陚���
@~{�z�H���CL����
@��R�?\)���HCn                                    Bxw�=�  �          @�\����@����c33��(�C  ����@�� ����z�C+�                                    Bxw�L:  �          @��H���R@�  �4z����C �3���R@�\)����PQ�B��{                                    Bxw�Z�  �          @�z��aG�@�ff�}p����B��q�aG�@���-p���\)B�u�                                    Bxw�i�  �          @�{�c33@�  ��33�	�
B���c33@����H�����RB�Ǯ                                    Bxw�x,  �          @����k�@�  ��33��=qB��\�k�@�  �5��33B�                                    Bxw���  �          @��z�H@����p���B�\�z�H@�(��?\)���B�u�                                    Bxw��x  �          @��R�z=q@���������B�B��z=q@��=p���
=B���                                    Bxw��  �          @�ff�l(�@��H��ff�Q�B��f�l(�@�p��QG����B��H                                    Bxw���  T          @����e�@�  ������B���e�@�33�Vff��=qB��H                                    Bxw��j  �          @����(�@�{��  ���C�=��(�@�\)�Fff��  B��q                                    Bxw��  �          @�����R@�z������	{C�q���R@��R�P�����B�Ǯ                                    Bxw�޶  �          @������
@�33��\)�p�C\)���
@�ff�U��\)B���                                    Bxw��\  �          @����{�@�{���\�  C���{�@���n{���
B�\                                    Bxw��  �          @����xQ�@�����p��(�C ^��xQ�@����aG��֣�B��3                                    Bxw�
�  �          @�  ���@�33���z�CL����@��R�\��\)B�Ǯ                                    Bxw�N  T          @�Q��|��@���  �33C(��|��@��\�h����=qB��=                                    Bxw�'�  �          @����|��@����(��\)C�)�|��@���b�\��
=B�z�                                    Bxw�6�  �          @��~�R@�����33�{C!H�~�R@��aG���
=B�k�                                    Bxw�E@  �          @��
�q�@�
=����ffC ���q�@��H�`  ���B�\                                    Bxw�S�  �          @�\�u@�=q�����HC�u@��R�dz���G�B�33                                    Bxw�b�  T          @�\�b�\@�=q�����!�B��\�b�\@�Q��s�
��\)B�p�                                    Bxw�q2  �          @�\�^�R@��
��z��  B�
=�^�R@�  �_\)��G�B�Q�                                    Bxw��  �          @�(��c�
@�(�����ffB�8R�c�
@����`  �ڏ\B�ff                                    Bxw��~  �          @��H��\)@[����R�.��C���\)@�
=��33�  Ch�                                    Bxw��$  �          @�\)��ff@g����R�'\)C0���ff@�33��=q��CY�                                    Bxw���  �          @����p��@������  C�\�p��@���fff����B���                                    Bxw��p  �          @���C�
@�����p�B�  �C�
@��\�P���ҏ\B�ff                                    Bxw��  �          @�{��p�@��R��\)��HB����p�@Ǯ�=p���z�B��                                    Bxw�׼  �          @���?���@�z��p�����B��?���@ᙚ�33����B�Ǯ                                    Bxw��b  �          @�R?��@�=q���R��
B�.?��@�z��Fff�ģ�B�ff                                    Bxw��  �          @�{?�  @\��p��\)B�B�?�  @ڏ\�0����Q�B�L�                                    Bxw��  �          @�p�?��@��w����B�  ?��@��
�(���{B�W
                                    Bxw�T  �          @�33?��@��H���H�%
=B��H?��@�Q��e���33B�                                    Bxw� �  �          @�\)?�=q@����(���\B���?�=q@ָR�@  ���B���                                    Bxw�/�  T          @�
=?��@��������  B�?��@�z��J=q��ffB�8R                                    Bxw�>F  �          @�p�?O\)@��\�����!�B�z�?O\)@Ϯ�]p���33B��q                                    Bxw�L�  �          @�G�?��
@�(���  ��B�L�?��
@�
=�K�����B��                                    Bxw�[�  �          @�\��@�������${B�\��@����_\)��ffB���                                    Bxw�j8  �          @�z�8Q�@�����\)��B����8Q�@�G��X����33B��                                    Bxw�x�  �          @��?��
@�G�������B�
=?��
@�p��U�ָRB��                                    Bxw���  �          @�33@   @�G���p��33B�� @   @�p��Z=q�ݙ�B�
=                                    Bxw��*  �          @�33?�  @�G���p��
=B��H?�  @��
�G
=�ɮB�8R                                    Bxw���  �          @��
?��
@�ff���B���?��
@У��E��  B���                                    Bxw��v  �          @�33?�\)@˅�)����{B�33?�\)@��ÿ�����\B�#�                                    Bxw��  �          @�ff>L��@��
��33�#�B��H>L��@�G��`���ᙚB�u�                                    Bxw���  �          @�Q�?+�@˅�z�H��{B���?+�@���(���B�(�                                    Bxw��h  �          @�  ?O\)@�����33�G�B��R?O\)@��H�;���ffB��                                     Bxw��  T          @�G�?�p�@�ff��ff��B��R?�p�@����B�\����B��                                     Bxw���  �          @�
=?��
@�(���p��(�B�u�?��
@ָR�A���33B�ff                                    Bxw�Z  T          @�?�
=@������H���B�?�
=@����R�\��33B��H                                    Bxw�   �          @�  ?޸R@��R��{�"
=B�#�?޸R@Å�\(���33B�L�                                    Bxw�(�  �          @�?��@�����H�0Q�B��?��@����u���HB��R                                    Bxw�7L  �          @�\?���@��R��33�>=qB��3?���@�����{�  B�=q                                    Bxw�E�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxw�T�   W          @�Q�?�@+�����\Bx�?�@���(��]=qB�z�                                    Bxw�c>  �          @�\)?�{@�����33�0�B�  ?�{@ʏ\������HB�aH                                    Bxw�q�  �          @��R?�(�@��
����.p�B�\)?�(�@���\)���B�#�                                    Bxw���  �          @�p�@%�@�(������G��Bg�@%�@����  �33B��                                    Bxw��0  �          @�ff@33@��R��=q�=Q�B��@33@�=q��z���RB�8R                                    Bxw���  �          @�ff?�@��
�*�H��=qB��)?�@��ÿ��
����B�\)                                    Bxw��|  T          @�Q�?5@�33�c33��ffB�33?5@�
=�����h��B�(�                                    Bxw��"  �          @�=q?5@ڏ\�L����B�{?5@��
�����C�B���                                    Bxw���  �          @��?�\)@����G��p�B��?�\)@ۅ�5���B��                                     Bxw��n  �          @�ff?O\)@������H�S�B�?O\)@�����{�!�B�\                                    Bxw��  �          @�  ?\)@��H��ff�U{B�\?\)@�������!�B�\                                    Bxw���  �          @�
=?�G�@Y���ڏ\�z�B���?�G�@��H�����H{B�u�                                    Bxw�`  �          @���?�z�@������33Be\)?�z�@mp��˅�h
=B�ff                                    Bxw�  �          @��H?W
=@.{�׮8RB��)?W
=@�����R�ZG�B�G�                                    Bxw�!�  �          @陚?}p�@0  ��z�u�B��{?}p�@�p���33�W=qB�aH                                    Bxw�0R  �          @�>�G�@R�\�أ��ffB��H>�G�@�\)����K�B�p�                                    Bxw�>�  �          @�
=?��@|(������g�HB��?��@�����\)�5p�B��                                    Bxw�M�  �          @��>\@�����{�N(�B�>\@�����\)B��)                                    Bxw�\D  �          A{���@�����z���\B�����@�G��L(����HB��                                    Bxw�j�  �          A Q��L(�@���`  �Џ\B���L(�@��ÿ�(��d��B���                                    Bxw�y�  �          @�=q�{@�  �AG���ffB�.�{@�  ���(  Bُ\                                    Bxw��6  �          @����@��
�C33��=qB�G���@���ٙ��N{B�G�                                    Bxw���  T          @�z��l(�@�(�<��
>\)B�(��l(�@θR?\A8��B�z�                                    Bxw���  T          @���.{@ڏ\��\�\)B�ff�.{@أ�?�ffA�\B�                                    Bxw��(  �          @��H�#33@�Q쿘Q���B�L��#33@ۅ>�p�@:=qB�Ǯ                                    Bxw���  �          @���L��@�zῠ  �ffB� �L��@�Q�>�z�@��B�                                    Bxw��t  �          @����z=q@��
����.{B���z=q@ȣ�<�>�  B���                                    Bxw��  T          @�
=�z=q@���˅�DQ�B��H�z=q@˅�\)����B�=q                                    Bxw���  �          @�
=�^�R@��
��
=�Q�B�G��^�R@�\)>u?��HB�k�                                    Bxw��f  T          @�{�Vff@�Q�?�z�A�  B�W
�Vff@�p�@J�HA�{B��
                                    Bxw�  �          @�ff�c33@���?E�@ϮB��)�c33@�
=@�A�p�B��f                                    Bxw��  �          @�=q��(�@��þ���=qC ���(�@�  ?@  @˅C 0�                                    Bxw�)X  �          @�=q�w
=@���\)�9�B��
�w
=@�33��Q�@  B�.                                    Bxw�7�  �          @���e�@�  ���
�
�HB��
�e�@��\>��R@'�B�{                                    Bxw�F�  �          @�Q��`��@��������B�\)�`��@��>��
@*�HB��                                    Bxw�UJ  �          @�z��_\)@�=q�@���У�B��_\)@�(���G��n�RB�\                                    Bxw�c�  �          @ٙ��mp�@Z�H�Q��ɅC	T{�mp�@xQ�\�}G�C�                                     Bxw�r�  �          @�
=��33@.�R?�(�Ao
=C���33@  @�
A�p�C�                                    Bxw��<  �          @��
��=q?s33@���B��C+{��=q�#�
@��B	��C4�                                    Bxw���  �          @���33?aG�@�=qB
�C+c���33���
@��B��C4��                                    Bxw���  �          @�
=���?p��@~�RB
�C*�=����#�
@��HB\)C4\                                    Bxw��.  �          @�
=��\)?��\@y��B(�C(���\)>���@��HB	=qC0�H                                    Bxw���  �          @�Q�����?�ff@�  BG�C)n����=�Q�@�(�B�RC3
=                                    Bxw��z  �          @�  ��G�?333@��B�HC,�=��G����R@�G�B��C7+�                                    Bxw��   �          @�  ��Q�?���@��B�C(����Q�=���@��B  C2��                                    Bxw���  �          @ᙚ��G�?�@p  B �\C&:���G�?   @~{B	p�C/                                    Bxw��l  �          @�\��\)@��@S33A�{C!����\)?��@mp�A�  C(�\                                    Bxw�  �          @��У�?�p�@Y��A��
C#!H�У�?���@p��A�
=C*33                                    Bxw��  �          @�  ��z�?��
@g�A�33C&���z�?#�
@xQ�A�p�C.L�                                    Bxw�"^  �          @�ff��Q�?˅@l(�A홚C%����Q�?.{@}p�B �C-�=                                    Bxw�1  �          @����(�?�{@}p�B�C%ٚ��(�>Ǯ@�p�B�
C/�f                                    Bxw�?�  �          @�Q�?+�@|�����
�J(�B��?+�@���s33�33B���                                    Bxw�NP  �          @����hQ�@5�>�=q@Y��C{�hQ�@+�?n{A;
=C��                                    Bxw�\�  �          @�
=��ff?��R@G�A���C".��ff?���@`  A�(�C)�                                    Bxw�k�  �          @��H�Ǯ?�\)@^{A��C%k��Ǯ?@  @p��A�33C-�                                    Bxw�zB  �          @�=q��
=?k�@���B  C+@ ��
=��\)@�z�B
�C4�H                                    Bxw���  �          @�{���R�W
=@���BQ�C<�q���R���@�ffB��CG)                                    Bxw���  �          @�\)���?5@�(�B=qC,�������Q�@�B�C7�H                                    Bxw��4  �          @�������?��H@�33B33C%������>�G�@��\B\)C/��                                    Bxw���  �          @�\)��  ?��H@�33B(�C'�f��  >B�\@���B�C2                                    Bxw�À  �          @�z���Q�?#�
@U�B
�HC,T{��Q�.{@X��B��C6�                                    Bxw��&  �          @����{��R@c33BQ�C;�3��{����@R�\B�CF)                                    Bxw���  �          @ָR��G�����@�G�BffCE����G��!�@a�B G�CN�H                                    Bxw��r  �          @�(������@w
=B=qCMǮ����L(�@K�A�
=CU}q                                    Bxw��  �          @�����@�
?���A��
C�f���?��
@��A�=qC$�)                                    Bxw��  �          @�G��8Q�@����g
=��HB��8Q�@�\)�
�H����B�B�                                    Bxw�d  �          @�=q���@�33��(��33B�#׾��@�{�-p�����B�                                    Bxw�*
  �          @ҏ\>k�@����Vff����B�W
>k�@ə��������B���                                    Bxw�8�  �          @ٙ�>�Q�@��R�������B�W
>�Q�@ȣ��%���B�W
                                    Bxw�GV  �          @��?�@�ff�qG��
p�B�\?�@�ff����RB�aH                                    Bxw�U�  �          @��H?z�H@����r�\�z�B���?z�H@��������
B��{                                    Bxw�d�  �          @���?��@��H�s�
�{B�
=?��@Å�����
=B��q                                    Bxw�sH  �          @��
?�G�@���p���G�B�ff?�G�@�=q�=q��33B�L�                                    Bxw���  �          @��?Ǯ@�=q��p��#G�B�G�?Ǯ@�ff�:�H��=qB�
=                                    Bxw���  �          @���?�(�@{���
=�6  B}Q�?�(�@���W��(�B�.                                    Bxw��:  �          @�
=?�ff@���P  � �B��?�ff@��R��33��G�B��                                    Bxw���  �          @���?^�R@�z�������RB��?^�R@��ÿO\)��
=B��f                                    Bxw���  �          @�(�?���@�\)�!G����B�.?���@�p��xQ���RB���                                    Bxw��,  �          @���?�(�@�ff�(Q���z�B��H?�(�@�p�����$��B��                                    Bxw���  �          @��
?��R@�(��A����B�k�?��R@��R��{�eB��                                    Bxw��x  �          @ҏ\?��@�{�N�R��33B���?��@�=q���
�{
=B�(�                                    Bxw��  �          @��H?��@�z��1G�����B���?��@�z῔z��z�B�G�                                    Bxx �  �          @��H>��H@�\)�����B���>��H@�녾����FffB�Q�                                    Bxx j  �          @��?(�@�z������  B�(�?(�@�\)����G
=B���                                    Bxx #  �          @�
=?�@���
=����B��R?�@�������B�.                                    Bxx 1�  �          @�=q=��
@�
=��\����B���=��
@�녾���HQ�B��R                                    Bxx @\  �          @�Q�Y��@�p���p��t��B��{�Y��@�{���aG�B��                                    Bxx O  �          @�Q�?��@�{��\)�-�B���?��@��
�B�\���HB�z�                                    Bxx ]�  �          @�\)?\(�@�Q��\�����B��q?\(�@�{��=q�s
=B�\)                                    Bxx lN  �          @�\?�z�@�Q��^{��RB���?�z�@�{���qG�B�B�                                    Bxx z�  �          @�\)��@�R��
=�o\)B�Q��@�R�L�;�33B�\                                    Bxx ��  �          @��ÿB�\@�{����z�B�#׿B�\@��B�\��p�B���                                    Bxx �@  �          @�  ��G�@����p��uG�B�\��G�@����ͿG�BÀ                                     Bxx ��  �          @�  ���@���33�+�
B�p����@�
=?�@|��B�W
                                    Bxx ��  �          @�\)��(�@��Ϳ}p����HB��3��(�@��?n{@�(�B��                                    Bxx �2  �          @�
=�(�@��Ǯ�@��B�Q�(�@���?\A;�B��                                    Bxx ��  �          @�R�W
=@�z�k����
B��W
=@�R?�AN�RB�W
                                    Bxx �~  �          @�=q����@�ff>u?�{B��H����@���@z�A�{Bƣ�                                    Bxx �$  �          @��L��@�p�?�=qA2�HB����L��@Å@@  A�p�B��
                                    Bxx ��  �          @�33�8Q�@�녿W
=��B�\�8Q�@��?O\)@陚B�
=                                    Bxxp  �          @�\)���@�G�@8��A�  C0����@i��@}p�B
��Cs3                                    Bxx  �          @����(�@�33@*=qA��C\��(�@�  @tz�B(�C	��                                    Bxx*�  �          @��H��p�@�(�@?\)AǮC����p�@|(�@���Bp�C�)                                    Bxx9b  �          @�Q��e@�G�?�{A��B�k��e@��
@J=qA��HB��3                                    BxxH  �          @��Q�@�Q�>�(�@n�RB�aH�Q�@�ff?�z�A�B�#�                                    BxxV�  �          @�=q�L(�@��
��\��33B���L(�@ȣ׿.{����B�
=                                    BxxeT  �          @�33�S33@�=q����
=B�=q�S33@�
=�@  ��  B���                                    Bxxs�  �          @�����H@���@\)A��C ����H@�  @g
=A�(�C^�                                    Bxx��  �          @�33���H@��?��RA<  CaH���H@�\)@5�A��C��                                    Bxx�F  �          @��H���\@�
=@33A�  B�����\@��@^�RA��HCE                                    Bxx��  �          @��
���@�(�@Q�A��\B�8R���@��@qG�A�ffC}q                                    Bxx��  �          @���{�@���?�Au�B�8R�{�@��H@XQ�AݮB�L�                                    Bxx�8  �          @�Q��mp�@�Q�?�33AS
=B�Q��mp�@��@K�A�{B�Q�                                    Bxx��  �          @�=q�w�@�\)?��AB�HB��
�w�@��@C�
A���B��                                    Bxxڄ  �          @�(��p  @���@�A\)B����p  @���@c33A�(�B��)                                    Bxx�*  �          @�z��tz�@�G�?�Ag
=B����tz�@��\@W�A��
B�8R                                    Bxx��  �          @������@�33?�ffAaB��f���@��@R�\A�\)C ��                                    Bxxv  �          @���p  @�=q?�z�Ap(�B�ff�p  @�33@]p�A�p�B�#�                                    Bxx  �          @�ff�i��@�33@p�A�  B�R�i��@�G�@p  A�ffB�#�                                    Bxx#�  �          @��
�l(�@���?���AG�B�q�l(�@�p�@HQ�A�(�B��\                                    Bxx2h  �          @��
�s33@�=q?�G�A>�\B�
=�s33@�{@E�A�33B�Ǯ                                    BxxA  �          @�p��}p�@��@7�A���B��)�}p�@��H@�Q�Bz�C�=                                    BxxO�  �          @�\��=q@���@#33A���B�Ǯ��=q@�(�@|(�B=qCJ=                                    Bxx^Z  �          @�����@���@$z�A���B��H����@��
@}p�B=qC�                                    Bxxm   �          @�(���G�@��\@0  A�G�C �
��G�@�z�@��HB�RC\)                                    Bxx{�  �          @�\)���
@���@?\)A���C�H���
@���@���B�C�f                                    Bxx�L  �          @�{���@���@1G�A���C���@�33@�33B��CaH                                    Bxx��  �          @�  ��33@�(�@  A�
=C ����33@��\@c�
A�z�C}q                                    Bxx��  �          @��
�xQ�@��
>�@uB�Q��xQ�@���?�(�A��
B���                                    Bxx�>  �          @��\)@�G�?
=@��RB�u��\)@�p�@�A��B�G�                                    Bxx��  �          @�z��|��@�z�?�33A=B��=�|��@���@4z�A�\)Cz�                                    Bxxӊ  �          @У��N�R@��?�{A!��B�3�N�R@���@!G�A�{B��f                                    Bxx�0  �          @�G�����@�������6ffB��f����@��H?���AL��B�k�                                    Bxx��  �          @�{��
=@���?c�
AffB�8R��
=@�@�RA�(�B�#�                                    Bxx�|  �          @ٙ���R@���@ ��A��B�ff��R@��
@�  B�HB���                                    Bxx"  �          @����S�
@���B�\��{B��S�
@���?�A6ffB��=                                    Bxx�  �          @��
��
=@��R@G�A�z�Bޅ��
=@�z�@c33Bp�B�                                      Bxx+n  �          @�(��!�@�?�@��B�(��!�@��\?�
=A��B�\)                                    Bxx:  �          @�G��7
=@��H@
=qA�Q�B�33�7
=@�G�@Z�HB
z�B��                                    BxxH�  �          @����\)@���@%�A���B�\�\)@w
=@s33Bz�B��                                    BxxW`  �          @�z��p�@�33@�AÅB�=q�p�@p  @c�
B�B�k�                                    Bxxf  �          @�����@�G�@�A��HB����@�Q�@Z=qB
��B�R                                    Bxxt�  �          @�����H@��
��  ��Bܞ����H@�������z�B���                                    Bxx�R  �          @�Q쿣�
@�
�s33�]
=B�𤿣�
@Q��@  �!��Bޅ                                    Bxx��  �          @�\)�W
=>��
��ff�C:�W
=?��R�|(�.B���                                    Bxx��  T          @�\)���@\���9����B�G����@�(������B�{                                    Bxx�D  �          @�Q��.{@�
=�N{����B��.{@�ff��=q��ffB�8R                                    Bxx��  �          @�p��h��?���\)�}p�B�q�h��@A��`���?��B�W
                                    Bxx̐  �          @��
=u?��
��ffB��{=u@1G����H�p�B�z�                                    Bxx�6  �          @�>\)�.{��\)¯��C�]q>\)?�(���G�#�B��                                    Bxx��  �          @�(��8Q�?�G���ff�{B�.�8Q�@:=q��33�s  BϸR                                    Bxx��  �          @�{��{@(���G��|�B�\��{@vff���
�>{B�B�                                    Bxx(  �          @�(���ff@,����\)�}�
B����ff@�����\)�;�B��                                     Bxx�  �          @�33����@��H��G��5{B�𤿙��@\�\(����B�ff                                    Bxx$t  �          @�\)�˅@��
��{�%��B��ÿ˅@���K���{B�8R                                    Bxx3  �          @�  ��p�@�Q�����)��B�uÿ�p�@�  �U��ҸRB��H                                    BxxA�  �          @�z��@�33�����)ffB�uÿ�@�33�XQ���ffBҳ3                                    BxxPf  �          @�{��
=@�z���33���Bٽq��
=@���AG����RBӣ�                                    Bxx_  �          @���\)@Å����
Q�Bѽq��\)@�������Bͳ3                                    Bxxm�  �          @�R��@��
�����2  B��f��@���`����ffB֊=                                    Bxx|X  �          @�׿�G�@�\)��33�*��B�(���G�@Ϯ�Vff���B�Q�                                    Bxx��  �          @��Ϳ�@�\)���
���B؊=��@��H�4z���G�BҸR                                    Bxx��  �          @��H��
=@�����33���B�\��
=@ָR��R��B�#�                                    Bxx�J  �          @�R���@�=q��=q���B�  ���@�33�$z����B�B�                                    Bxx��  �          @�
=�#�
@�(��S�
��{B�
=�#�
@��H��
=�7\)B�u�                                    BxxŖ  �          @������@�
=�3�
���B�W
���@�G���  ��Bݔ{                                    Bxx�<  �          @�
=�8Q�@�33�aG����B�p��8Q�@��?O\)@�33B�\)                                    Bxx��  "          @��R�G�@�p��#�
��G�B��G�@�{?�G�Ay�B�p�                                    Bxx�  �          @�{�-p�@�33�@  �p�B����-p�@�(�?�@��B�                                     Bxx .  �          @��R�.�R@�ff=L��?�B�33�.�R@�
=?���Av�HB�3                                    Bxx�  �          @����/\)@�������33B���/\)@�(�?�p�AUB�                                    Bxxz  �          @�\)���H@��\@G�A���Bׅ���H@�\)@`��B�HBݨ�                                    Bxx,   �          @��׿��@��\@�A���BҊ=���@��@o\)B!�B��                                    Bxx:�  �          @��Ϳ�ff@���?��A�p�B�k���ff@�
=@a�BB۸R                                    BxxIl  �          @�  ��@��
@1�A���B�Q��@���@�p�B#�Bݣ�                                    BxxX  �          @˅�޸R@��@+�A�G�B�.�޸R@�{@�ffB%��B��H                                    Bxxf�  �          @˅��@�@��A���B�p���@��R@z�HB33B�\                                    Bxxu^  �          @����@�\)?���A��
B�.��@���@[�B��B��                                    Bxx�  �          @\�{@�33?�ffA�Q�B����{@��@UBp�B��
                                    Bxx��  �          @��H�7�@�{?�\)AIG�B���7�@���@=p�A߮B��                                    Bxx�P  �          @�ff�1�@���?���A��HB�8R�1�@�\)@\��B ��B��                                    Bxx��  �          @�(����@��@s�
BG�B�Ǯ���@�p�@��BE��B�                                      Bxx��  �          @�G�����@��@\)B
��BӀ ����@��\@�  BN{B�\                                    Bxx�B  
�          @׮���@�G�@�Q�B�B�����@dz�@��BV��B�aH                                    Bxx��  �          @�  �u@��
@�ffB-\)B�k��u@A�@���Brp�B�                                      Bxx�  �          @��H�fff@��H@>�RAǮB���fff@��
@���B*p�B��f                                    Bxx�4  �          @�33�^�R@�
=@W�A�z�B�LͿ^�R@��
@�{B4
=B�B�                                    Bxx�  �          @�\��{@�
=@HQ�A�z�B˳3��{@�{@�
=B+(�B�L�                                    Bxx�  �          @��H���@��@<(�A��B�(����@��\@�=qB$�B�G�                                    Bxx%&  �          @�\)�ff@��
@%�A�G�B��ff@�  @�  B�HB�8R                                    Bxx3�  �          @�R���R@��@,��A�B�Q���R@��@�(�B�B��)                                    BxxBr  �          @���@�33@Q�A��Bم��@��H@���BB�aH                                    BxxQ  �          @��  @�p�@�\Az{B�\�  @�{@��HB
=B݅                                    Bxx_�  �          @�{��@�  @�A���B�#���@���@��\B��B�{                                    Bxxnd  �          @�p���R@�\)?�33Amp�B�p���R@���@z=qA�Q�B�L�                                    Bxx}
  �          @�R�)��@أ�?�33AL��B޽q�)��@��@l(�A�33B�G�                                    Bxx��  T          @����'
=@�
=?���AffB����'
=@ȣ�@N�RA˅B��                                    Bxx�V  �          @�Q��	��@�>�z�@  B�L��	��@�
=@"�\A�=qB�k�                                    Bxx��  �          @��z�@��#�
���
B��
�z�@ڏ\@{A�
=Bُ\                                    Bxx��  �          @�  �-p�@�ff�O\)��p�Bފ=�-p�@�33?�
=A/�
B��                                    Bxx�H  �          @���/\)@�\)�n{�ᙚB��H�/\)@��?�=qA"{B�=q                                    Bxx��  �          @�z��&ff@�녿�(���B����&ff@�33?�G�@��\Bݞ�                                    Bxx�  �          @���@�ff��Q��V{Bڀ �@�(�?�@��Bم                                    Bxx�:  �          @�33� ��@أ׿���$��Bܣ�� ��@��H?h��@�z�B�G�                                    Bxx �  �          @�=q�Q�@��H�����B��Q�@��=�?xQ�B�
=                                    Bxx�  �          @��8��@�33�^�R��ffB��8��@أ�?�{A(  B�=q                                    Bxx,  �          @�
=�C�
@����\)�)��B�\)�C�
@�  ?Tz�@�B�Ǯ                                    Bxx,�  �          @����=p�@��
��Q��3\)B�=q�=p�@�\)?B�\@�{B�=                                    Bxx;x  T          @�{�@��@�G������"�\B�Ǯ�@��@���?�\)Ai��B�k�                                    BxxJ  �          @�(��4z�@Ӆ��{�I�B�.�4z�@���?(�@��B�.                                    BxxX�  �          @�\)�(Q�@�ff��Q��x��B�Q��(Q�@�\)>�  ?�(�Bޣ�                                    Bxxgj  �          @���@����z��qB����@�p�>�33@,��BڸR                                    Bxxv  �          @�ff��@��H�����f�\B�����@�=q>��@fffB׮                                    Bxx��  �          @�z��33@��H�����RB�  ��33@���>��?�p�Bѽq                                    Bxx�\  �          @��H�z�@��Ϳ����)�B�\)�z�@޸R?xQ�@��
B�{                                    Bxx�  �          @����<(�@У׿�G��>�\B�=�<(�@��?333@���B��                                    Bxx��  �          @���W
=@��Ϳ��R�:�\B�k��W
=@�G�?0��@��\B�p�                                    Bxx�N  �          @�(��-p�@ڏ\���R��B�=q�-p�@љ�?�Q�Aup�B��H                                    Bxx��  �          @����(�@�33=�?h��B�33�(�@�@
=A��B݊=                                    Bxxܚ  �          @�G���
=@�G�?=p�@��B��f��
=@���@AG�A�{B�u�                                    Bxx�@  �          @����*�H@��>���@!G�Bݙ��*�H@У�@(Q�A�B��                                     Bxx��  �          @����-p�@�  >\@7�B�Q��-p�@�\)@*�HA�z�B�\)                                    Bxx�  T          @�{���@�녿�p��XQ�Bأ����@�  ?!G�@�(�B׸R                                    Bxx2  �          @��ÿ�z�@�Q��E���(�B��ÿ�z�@�{�h����  B��                                    Bxx%�  �          @�p��\)@�ff���tz�B��
�\)@�
=?��
Ae�B��                                    Bxx4~  �          @��;�@ƸR�
=��ffB�z��;�@���?�G�AJffB瞸                                    BxxC$  �          @�׿���@\�Mp����B͙�����@ٙ��}p��ffB���                                    BxxQ�  �          @߮���@�{�j�H���B�8R���@��H�Ǯ�O\)B�=q                                    Bxx`p  �          @�
=�8Q�@��\���
�J��B�  �8Q�@�(��g
=����B�
=                                    Bxxo  �          @�
=�
=q@`����
=�m33Bŀ �
=q@����ff�=qB���                                    Bxx}�  �          @�녿�
=@�����;B�aH��
=@�z��R�\��B���                                    Bxx�b  �          @���
=@�����
�z�B�.��
=@�
=�z���33B�(�                                    Bxx�  �          @��
�   @�G��\)�p�Bۨ��   @�녿���up�B��)                                    Bxx��  �          @���(�@�(��u��Bڅ��(�@�33���H�^�RB�8R                                    Bxx�T  �          @�(����@����;���  Bݏ\���@�ff�G����Bٳ3                                    Bxx��  T          @�
=�(Q�@˅�:=q��p�B��f�(Q�@�
=�����B�Q�                                    Bxxՠ  �          @��1�@�\)�1�����B�(��1�@ٙ��   �vffB��                                     Bxx�F  �          @���7�@�
=�.{����B�aH�7�@�G������B�ff                                    Bxx��  �          @�R�Q�@�z��=q�k\)B�.�Q�@�z�>\@C�
B�B�                                    Bxx	�  �          @��H�e�@��H�Tz�����B��H�e�@Ǯ?��A/
=B��                                    Bxx	8  �          @�  �tz�@����z����B�=�tz�@��
?�
=Ao
=B��f                                    Bxx	�  �          @����(�@��\?z�H@���C s3��(�@�(�@;�A�C
=                                    Bxx	-�  �          @��H���@�
=?n{@�G�B�����@���@@��A�33C.                                    Bxx	<*  �          @��
��p�@�Q�?fff@��HB�k���p�@�=q@@��A�{C                                      Bxx	J�  �          @����
@�?h��@��
C @ ���
@�\)@@  A��
C�H                                    Bxx	Yv  �          @�\)��\)@ʏ\?��
@�B����\)@��\@J=qA��CG�                                    Bxx	h  �          @�z���p�@�33=���?8Q�B�����p�@�p�@33A��
C =q                                    Bxx	v�  �          @����
=@˅?!G�@��B�{��
=@�  @3�
A���CaH                                    Bxx	�h  �          A z���G�@���@UA���Cz���G�@b�\@�(�B��C33                                    Bxx	�  �          A ����  @��
@e�A�=qC���  @q�@�\)B#  C�                                    Bxx	��  �          A ����(�@�
=@6ffA�=qC}q��(�@��@�B33C
B�                                    Bxx	�Z  �          A Q���(�@�{@333A�{C����(�@��H@��
B(�C
W
                                    Bxx	�   �          A z���ff@��@O\)A�
=B��f��ff@���@�=qBz�C�q                                    Bxx	Φ  �          A (���G�@��@k�Aڣ�B�#���G�@�{@�Q�B-C��                                    Bxx	�L  �          A (��z=q@��H@u�A�RB����z=q@��@���B3p�C��                                    Bxx	��  �          @�\)�`��@�G�@y��A�G�B�\)�`��@���@���B933Cs3                                    Bxx	��  �          @�\)�c�
@���@fffA��B�(��c�
@�
=@���B0��C �=                                    Bxx
	>  �          @�ff����@Ǯ@333A���B������@��H@��\B�
C�f                                    Bxx
�  �          @�
=�s33@θR@*=qA�z�B����s33@��H@�G�B  B��\                                    Bxx
&�  �          @��R���\@���@ ��Ai��B�����\@�z�@�ffA��B�8R                                    Bxx
50  T          @�\)����@��@�\A��C:�����@�G�@�{A��
C	�                                    Bxx
C�  �          @�{��{@�(�@!G�A�ffCT{��{@j=q@�33A�C:�                                    Bxx
R|  �          @��
��(�@��\@ ��AmG�C���(�@�
=@~{A�\C�                                    Bxx
a"  �          @�33����@��H@   A��HC������@��H@�  B�C��                                    Bxx
o�  �          @����  @��H?�  AMG�C�{��  @�33@c33A��HC�H                                    Bxx
~n  �          @�33��(�@��?޸RAN{C�{��(�@��\@o\)A�ffCL�                                    Bxx
�  �          @�(����@�{?�p�Ah��C�f���@��H@y��A��C	�=                                    Bxx
��  T          @�p�����@�=q@$z�A�G�C�q����@���@�=qB33C�)                                    Bxx
�`  T          @�z���z�@��@�A�{C \)��z�@��\@�z�B�HC\)                                    Bxx
�  T          @�z���G�@�G�@:�HA�=qC����G�@w�@�z�B�RCu�                                    Bxx
Ǭ  �          @�33��33@�p�@=qA�\)C���33@���@��Bp�C	�                                     Bxx
�R  �          @�������@�z�@C33A�\)C������@y��@��Bp�C�)                                    Bxx
��  T          @����p�@�Q�@L(�A��RCB���p�@}p�@�\)B��C��                                    Bxx
�  �          @����o\)@Ϯ@�A|z�B��)�o\)@�Q�@�33B�
B��q                                    BxxD  �          @�����R@���@�A�\)C�����R@���@�\)Bp�CB�                                    Bxx�  �          @�=q���\@�z�@#�
A��B�(����\@���@�z�B�
C�                                    Bxx�  �          @�=q��33@��H@>�RA��C	ٚ��33@Y��@��B��C�=                                    Bxx.6  �          @�������@��@+�A���C�{����@|��@�ffB�C                                    Bxx<�  �          @�����=q@��R@/\)A�33C.��=q@u�@�
=B��C��                                    BxxK�  T          @�������@�(�@>{A��Cu�����@j=q@���B(�C�                                    BxxZ(  
�          @�����  @�
=@4z�A��\C����  @s33@��B�
C��                                    Bxxh�  �          @����p�@�(�@7
=A�{C )��p�@�p�@�  B{C�                                    Bxxwt  T          @������@���@>{A�=qC�
���@Fff@�ffB��C�f                                    Bxx�  �          @�\)���@�(�@8Q�A�33C	���@]p�@��B\)C��                                    Bxx��  �          @�Q�����@��@@  A��C	������@U@��\B�C�\                                    Bxx�f  �          @�Q�����@��@QG�A�\)C������@4z�@��B�HC��                                    Bxx�  �          @�  ���H@�z�@I��A�Q�C!H���H@Vff@�  B�
C��                                    Bxx��  �          @�Q���(�@�(�@`  A��C	���(�@>{@��B G�C�                                    Bxx�X  �          @�
=��
=@�(�@h��A��C���
=@+�@���B"G�C�=                                    Bxx��  �          @��R��33@�@j=qA�RC�q��33@\)@�
=B ffC
=                                    Bxx�  �          @����{@��\@z�A�G�C
aH��{@fff@~{A��RC��                                    Bxx�J  �          @�p����
@���@��A�  C� ���
@s33@�  B p�Ck�                                    Bxx	�  T          @�����@�z�@
=A�ffC �=����@�=q@�=qB
ffCY�                                    Bxx�  �          @��H���@��H@333A��B�Q����@�33@��RBz�Ck�                                    Bxx'<  �          @�����@�  @"�\A�  C
=���@x��@�33BffC��                                    Bxx5�  �          @��Y��@�z�@%�A��B��Y��@�{@�\)B�B���                                    BxxD�  �          @��4z�@��?���Aep�B����4z�@�{@�  B
�B���                                    BxxS.  �          @����@Z=q@vffA��
C�\���?�@��B#�C"��                                    Bxxa�  �          @�\)��
=@�
=@A�p�C  ��
=@l(�@�=qBG�Cc�                                    Bxxpz  �          @�=q�p  @��R������B��=�p  @�?��HA!��B���                                    Bxx   �          @�Q��{�@��?��A(�B����{�@��\@C�
A�
=C��                                    Bxx��  �          @�R�1G�@�33?O\)@�(�B�.�1G�@���@L(�A��
B��                                    Bxx�l  �          @���ff@��<�>��C�R��ff@�(�@�A�CW
                                    Bxx�  �          @�  �Tz�@�z�@8��A�Q�B�  �Tz�@��H@�33B*p�C�                                    Bxx��  �          @��tz�@��@Q�A��
CE�tz�@0  @��RB4  CO\                                    Bxx�^  T          @������@�\)@p��A���C	
����@(�@��B2�
C��                                    Bxx�  �          @�=q��z�@aG�@W�A���C����z�?�
=@�
=B�
C L�                                    Bxx�  �          @�����H@��@'
=A�=qC
�)���H@G�@���B=qC��                                    Bxx�P  �          @������@�z�@L(�A�(�C �����@_\)@�{B+Q�C�                                    Bxx�  �          @������@��\?^�R@�\)C�R����@��\@2�\A�z�C@                                     Bxx�  �          @������R@�
=?��
A$��C�f���R@g
=@;�A�
=Cz�                                    Bxx B  �          @�G����R@�ff>�G�@^{C�����R@��@(�A�z�C	
                                    Bxx.�  �          @�����H@��H?��A33B�����H@��R@QG�A��
C}q                                    Bxx=�  �          @�G����@�  >�Q�@6ffB��R���@���@%A��HC��                                    BxxL4  �          @�  �{@���33�  B�  �{@˅?�p�AC33B�ff                                    BxxZ�  �          @�R�>{@�33�5����B���>{@Å?���AtQ�B�Ǯ                                    Bxxi�  �          @�\)�,(�@Ӆ�B�\��G�B�Q��,(�@��
?�Axz�B���                                    Bxxx&  �          @�p���z�@�녿���*=qB�aH��z�@أ�?�  AC�Bˀ                                     Bxx��  �          @���@�Q��=q�m��Bڨ���@�ff?h��@�G�Bٞ�                                    Bxx�r  �          @��Ϳ�\)@�{���\�,  B�ff��\)@��?��A<Q�Bԅ                                    Bxx�  �          @���
=@У׿�ff��RḄ׿�
=@���?��A`(�B��                                    Bxx��  �          @�p��6ff@�G��#�
����B�\�6ff@�  @&ffA��HB�Ǯ                                    Bxx�d  �          @�ff����@�G��!����B�ff����@�  >�z�@�B��H                                    Bxx�
  �          @�(���
=@�ff�S33��B��q��
=@�  �
=q��p�B��)                                    Bxxް  �          @�33��z�@ƸR�)�����B�  ��z�@�  =u>��BϮ                                    Bxx�V  �          @�33��=q@���������HB�#׿�=q@�z�?��@��RB�Ǯ                                    Bxx��  �          @��
���@�G�����(�B�\)���@�?   @���B�#�                                    Bxx
�  �          @����H@�(��7����HB�zῚ�H@߮�u���Bǣ�                                    BxxH  �          @�녿�@�p��(���  BȸR��@ۅ>�Q�@<��B�k�                                    Bxx'�  �          @�ff�+�@љ���(��`��B�u��+�@��@  A�{B��                                    Bxx6�  �          @���r�\@�{=�G�?fffB���r�\@��
@'
=A���B�                                      BxxE:  �          @�Q��s�
@���=#�
>���B�{�s�
@��@!�A���B�k�                                    BxxS�  �          @���`��@��H��G��_\)B����`��@�
=@
�HA�33B��                                    Bxxb�  �          @�\)�W
=@�z�=#�
>��
B�z��W
=@�=q@(Q�A�G�B�                                      Bxxq,  �          @��Ϳ�Q�@��H�:�H���
Bӳ3��Q�@�G�@�A��B�\                                    Bxx�  �          @�����@ָR���
=Bօ��@�33?�AY�B�\                                    Bxx�x  �          @�\��  @�ff?���A\)B��H��  @�
=@]p�A�  C ޸                                    Bxx�  �          @���\)@�p�?�p�Ab�\B��f��\)@�\)@s33B�RC�                                    Bxx��  �          @�Q��y��@���?�@�B����y��@���@=p�A��B��R                                    Bxx�j  �          @�=q��  @���?5@��
B�����  @�\)@H��A��
B��                                    Bxx�  �          @��o\)@��H?+�@�(�B���o\)@���@G�A�  B��                                    Bxx׶  �          @�ff�:�H@љ���  ���HB���:�H@�=q@{A�\)B�k�                                    Bxx�\  �          @�Q���Q�@���?z�@�33C	���Q�@�
=@#33A�p�CE                                    Bxx�  �          @�  ���R@��?�@��HC B����R@�{@3�
A���C
                                    Bxx�  �          @�\)��
=@Vff?��A((�C����
=@ ��@%�A�ffC�                                    BxxN  �          @�R��Q�@xQ�?k�@��HC&f��Q�@I��@��A���CW
                                    Bxx �  �          @���{@�=q?�R@�p�C�
��{@\��@��A�\)C�f                                    Bxx/�  �          @�G����@�\)?�G�AA�B��f���@��\@p��A�Q�C33                                    Bxx>@  �          @陚��Q�@�(�@{A��\B�����Q�@�{@��BC�                                     BxxL�  �          @�Q����@��H?��Ar�HC�����@u@uB �RC�q                                    Bxx[�  �          @�  ���H@���>�ff@g�B�ff���H@��\@5�A���C �
                                    Bxxj2  �          @�(��\@�G��8�����HB�B��\@�p�    <#�
B��
                                    Bxxx�  �          @�G��b�\@�\)��Q���HB�=q�b�\@��?��
ABffB���                                    Bxx�~  �          @�=q�mp�@ȣ׾��hQ�B�8R�mp�@�z�@p�A��B�u�                                    Bxx�$  �          @陚���\@�{�L���ʏ\B������\@�\)?�\Ab{B���                                    Bxx��  �          @���u�@�{��{�-B����u�@��R?��\A"�\B�z�                                    Bxx�p  �          @�Q����@љ������
B�����@ۅ?J=q@ǮB׊=                                    Bxx�  �          @���z�@ҏ\��\���B�k���z�@�?333@��HB��)                                    Bxxм  �          @�
=��=q@�G��2�\���RB��Ϳ�=q@�>k�?�\)B�W
                                    Bxx�b  �          @�ff��@����?\)��
=B�p���@�=L��>��B�ff                                    Bxx�  �          @�p�=u@�=q�4z���B��H=u@�z�>k�?�{B���                                    Bxx��  �          @��;��@�=q�,(����B�#׾��@��H>�Q�@8Q�B��=                                    BxxT  �          @�R��\)@�  �(�����B�\��\)@�p�?��@�p�B�33                                    Bxx�  �          @�ff�G�@��C�
���
B�.�G�@�z�\)��{Bԣ�                                    Bxx(�  �          @�R����@��H�I���ϮB˔{����@�=q�����
=B�.                                    Bxx7F  �          @�z��w�@g
=�����%=qC�3�w�@�  �*=q����B��R                                    BxxE�  �          @��H��ff@��������-  B�W
��ff@��
�33��33BӔ{                                    BxxT�  �          @�녿��@��������B�����@�{�˅�Q�B���                                    Bxxc8  �          @�=q�-p�@�33�y���z�B�aH�-p�@��������E�B�33                                    Bxxq�  �          @����O\)@�
=�i����
=B�B��O\)@�녿��{B�W
                                    Bxx��  �          @�  �G
=@�  ������B�W
�G
=@��׿�=q�RffBꙚ                                    Bxx�*  �          @�׿�p�@���u��
B��ÿ�p�@Ӆ������\B�W
                                    Bxx��  �          @ᙚ��  @�\)�aG���33B�{��  @�p��������BĊ=                                    Bxx�v  �          @�  �xQ�@����s33��HB��ÿxQ�@ۅ�n{��p�B�                                    Bxx�  T          @�׿W
=@����tz��ffB�ff�W
=@�(��p����{B��
                                    Bxx��  �          @�G��  @�z��`����=qBߙ��  @�33�8Q����BٸR                                    Bxx�h  �          @�׿�
=@�G��[���\B�Ǯ��
=@ָR�z���\)B�
=                                    Bxx�  T          @߮��
=@���c33��Q�Bʙ���
=@ۅ�#�
��
=Bǅ                                    Bxx��  �          @����
@�p��P  ���B�L���
@У׾��uB�                                      BxxZ  �          @��a�@�\)�Y���߮B�{�a�@�  ?�As�B�                                    Bxx   �          @����n{@���?!G�@��B�\)�n{@���@L��A��
B��3                                    Bxx!�  �          @������H@��>�\)@  B�����H@�
=@.{A�p�C�                                    Bxx0L  �          @�����@��
?�ffA'�C&f����@�Q�@`��A�C�                                     Bxx>�  �          @���(�@�  ?�z�A5�C ����(�@��\@k�A���CT{                                    BxxM�  T          @������
@�\)?��A-p�C �
���
@��\@g
=A���C:�                                    Bxx\>  �          @�33�G
=@��׿�G��h��B�{�G
=@�?��A�
B�\)                                    Bxxj�  �          @ᙚ�33@\�%����B�p��33@�33>�33@7
=B�p�                                    Bxxy�  �          @���Fff@�z�>���@(Q�B�=�Fff@�@0  A�  B�8R                                    Bxx�0  �          @�\)�Tz�@��?��
A)B�{�Tz�@��@r�\B��B�L�                                    Bxx��  �          @����e@�  �u���HB��
�e@��@%A�(�B�33                                    Bxx�|  �          @���Dz�@�  ��R���
B�W
�Dz�@�z�@��A��HB�#�                                    Bxx�"  �          @�
=�   @��Ϳ������Bޞ��   @ƸR?��Az�\B���                                    Bxx��  �          @�{��  @�=q��
���\BθR��  @�{?:�H@�=qB�G�                                    Bxx�n  �          @���\@�=q�0����  B�.��\@�z�>���@0��B�k�                                    Bxx�  �          @��
����@�ff�s33�z�B������@ٙ��^�R��G�B�z�                                    Bxx�  �          @ۅ���@���Z=q��B�#׾��@أ׾�ff�s�
B�p�                                    Bxx�`  �          @����@�  �G��ٙ�BՊ=���@�ff@A�  B�.                                    Bxx  T          @�=q�vff@�z�@9��A�=qB�.�vff@��R@��\B,G�Cz�                                    Bxx�  �          A{�~{@�  @p��A��B����~{@l(�@�G�B@��C	�                                    Bxx)R  T          @�\)��=q@��
@P��A��\B�����=q@s33@�G�B/�C
��                                    Bxx7�  �          A Q�����@��@7
=A��B�  ����@�z�@���B$Q�C                                    BxxF�  �          A����z�@�{@=p�A�\)C ���z�@~�R@��B#�RC^�                                    BxxUD  �          A����R@�@|(�A�\Ch����R@E�@��RB>�
C^�                                    Bxxc�  �          AG���Q�@���@�p�B\)C� ��Q�@33@�G�BX��CE                                    Bxxr�  �          A��G�@�  @���A�
=C�f��G�@4z�@�=qBBffC0�                                    Bxx�6  �          @��H���H@��\?�z�AbffC�����H@��H@��\B�\C
)                                    Bxx��  �          @�33��=q@��\?�Q�AMp�C ���=q@�{@�z�B�HC��                                    Bxx��  �          @������\@�\)?�p�A333C����\@�{@z=qA�Q�C	aH                                    Bxx�(  �          @��
��@��H?�=qA?33Cff��@���@{�A��RC\                                    Bxx��  �          @�=q��\)@���@�A��C���\)@tz�@�p�B��C}q                                    Bxx�t  �          @������
@�@]p�A��
C8R���
@C33@�B-��C�                                    Bxx�  �          @�
=����@�Q�@Z=qA��C�R����@H��@�p�B+��CW
                                    Bxx��  
�          @�p�����@�
=@I��A�C
����@N{@�B#\)C�                                    Bxx�f  �          @����{@��@H��A�p�C����{@S33@��RB%{C8R                                    Bxx  �          @�
=��{@��@L(�A�
=C����{@Tz�@���B&G�C
                                    Bxx�  �          @�
=���@��H@p��A���C����@4z�@��B4��C��                                    Bxx"X  �          @�\)���@�@��RA��RCff���@p�@��BB�RC!H                                    Bxx0�  �          @�
=���\@���@w
=A�{C�)���\@-p�@�\)B8  C��                                    Bxx?�  �          @�
=��@��@qG�A�(�C���@-p�@�(�B4�C33                                    BxxNJ  �          @�(����
@��H@`��A�(�C�����
@,(�@��HB*CG�                                    Bxx\�  �          @�\��33@��R@n�RA�  C	n��33@��@��B4�C{                                    Bxxk�  �          @�p���Q�@�p�@Tz�A�  C����Q�@=q@�
=B"�Ck�                                    Bxxz<  �          @����33@{�@fffA�\)C^���33?�@�  B*�C #�                                    Bxx��  �          @�z���  @��@Z=qAۮC�3��  @��@�\)B)G�C@                                     Bxx��  �          @�(���p�@�  @J�HA�\)C	����p�@"�\@�(�B%�RC�3                                    Bxx�.  �          @��
����@n{@���B�RC�=����?���@�G�B7�HC#�                                    Bxx��  �          @�����@��
@P��A�C}q��@	��@�G�B"{Cn                                    Bxx�z  �          @�{��z�@�@!G�A�Ck���z�@#33@��BC��                                    Bxx�   �          @�=q����@���@
=A�\)C5�����@3�
@��
B
G�C�R                                    Bxx��  �          @����{@��
@
=qA��C
O\��{@Fff@���BQ�C&f                                    Bxx�l  �          @陚���\@Y��@z�HB�\C=q���\?���@���B0
=C&�                                     Bxx�  �          @�  ���@��@�B
=C�{���=�@��B1�C2��                                    Bxx�  �          @�R��p�@Q�@��B�CQ���p�>L��@��B$�C1�                                    Bxx^  �          @�ff��  @(�@|��B�C���  >���@�(�B G�C0��                                    Bxx*  �          @�Q����@-p�@k�A���Cff���?0��@���Bz�C-
=                                    Bxx8�  �          @�\�Å@U�@  A�33Cu��Å?�33@b�\A�p�C"�                                    BxxGP  T          @�G���=q@8Q�@�{B�\C���=q>�Q�@��BB�C/�3                                    BxxU�  �          @�G����@]p�@���B��Cٚ���?��@�p�B@p�C'h�                                    Bxxd�  T          @�G����R@`  @j�HA���C{���R?���@�(�B(C%(�                                    BxxsB  �          @�G����@aG�@\��A��
C�3���?�  @�{B p�C$W
                                    Bxx��  �          @�33��=q@�(�@7�A���C.��=q@�
@�
=B�\C��                                    Bxx��  �          @�G�����@��H@$z�A�C	�H����@6ff@�p�BC}q                                    Bxx�4  �          @�Q���Q�@��\@!G�A��C	����Q�@7
=@��
B�RCE                                    Bxx��  �          @�\)��  @�(�@=qA��C	&f��  @=p�@��B33CY�                                    Bxx��  �          @�z�����@�ff@�A�G�C
� ����@1�@�\)B�\C�                                    Bxx�&  �          @�z����@�=q@p�A��C����@E@�B��Cn                                    Bxx��  �          @����R@���@\)A�ffC�����R@QG�@�=qB(�C�q                                    Bxx�r  �          @�Q���(�@�33@�HA�Q�C
B���(�@,��@�B�C
=                                    Bxx�  �          @�p���(�@i��@ffA�
=C޸��(�@
=@r�\B�C�)                                    Bxx�  �          @����@i��@ ��A�{C� ���@�\@|(�B
�C
=                                    Bxxd  �          @����@vff@��A���C�3���@��@{�B
�C�f                                    Bxx#
  T          @�\)��z�@u@��A�
=C�{��z�@@p  B(�C�\                                    Bxx1�  �          @�\)��z�@y��@z�A�{C!H��z�@��@j�HA�{C�                                    Bxx@V  �          @��
��Q�@~{@�A�z�C����Q�@@���B
=C�                                    BxxN�  �          @ڏ\��{@�?���A3�C!H��{@C33@I��A���C�                                    Bxx]�  �          @����33@�\)?+�@�z�C����33@vff@5A���C�\                                    BxxlH  �          @�=q��=q@�=q?�=qA5p�Cz���=q@X��@VffA���Cٚ                                    Bxxz�  �          @��H���@�\)?W
=@�33C�����@p��@@  A��C:�                                    Bxx��  �          @��H��(�@���>.{?��HC����(�@��\@"�\A��HC��                                    Bxx�:  �          @�33��  @�z�>�  @�C(���  @�@#33A��
C
��                                    Bxx��  �          @��
����@�>k�?��HC�����@�
=@#�
A���C
��                                    Bxx��  �          @��
���\@��H?&ff@��RC����\@|(�@8��Aƣ�C�=                                    Bxx�,  �          @�33��@��R?   @�ffCJ=��@y��@,(�A�p�C�f                                    Bxx��  �          @�(���
=@�\)>�(�@c33C^���
=@|��@(��A��\CxR                                    Bxx�x  �          @������H@��<��
>\)C�H���H@�Q�@A��C
�f                                    Bxx�  �          @ۅ���@�G��\)��z�C� ���@��R@=qA���C��                                    Bxx��  �          @��H��
=@�p���
=�aG�C����
=@���?��RA��CB�                                    Bxxj  �          @�ff��33@�{��\)�ffC����33@�{@\)A���CxR                                    Bxx  �          @޸R���\@��H�\(�����C���\@�33?�Ar=qCT{                                    Bxx*�  �          @�z���{@��\����Q�C &f��{@�
=?���AR�RC ��                                    Bxx9\  �          @�������@���+����C �q����@���?��RA��
C��                                    BxxH  �          @���{@���>B�\?ǮC�R��{@���@.{A�
=CJ=                                    BxxV�  T          @�ff��Q�@��?��A
=qC#���Q�@���@Y��A���C
G�                                    BxxeN  �          @�������@���?�z�A9p�C�)����@p  @j�HA��RC!H                                    Bxxs�  T          @�\���@�ff?�  AC�C	����@Y��@e�A�C)                                    Bxx��  �          @���G�@�p�?�{Ar�HC:���G�@Y��@�  B	(�C��                                    Bxx�@  T          @��H���R@�  @\)A�(�C	� ���R@7
=@�z�B��C
=                                    Bxx��  �          @޸R��p�@�z�?���A0z�C����p�@L��@Q�A���C:�                                    Bxx��  �          @�Q���z�@qG�?
=@�(�C���z�@@��@A�{C��                                    Bxx�2  �          @ڏ\��\)@u?��A��C�R��\)@��@`  A��C�                                    Bxx��  �          @�ff��p�@K�?���A��C����p�@  @ ��A�z�Cc�                                    Bxx�~  �          @�33���@�(�?n{@��\C�f���@fff@C�
Aՙ�C�q                                    Bxx�$  �          @����=q@���?�G�AK\)C���=q@G
=@]p�A�Cu�                                    Bxx��  �          @�p���=q@�?}p�AC
����=q@X��@AG�Aҏ\CE                                    Bxxp  �          @�{��\)@hQ쿓33�G�C����\)@n�R?E�@��C��                                    Bxx  T          @����(�@�(����
�.{C����(�@g�?��RA��RCE                                    Bxx#�  T          @����
=@�G��Q���\)C�)��
=@���?���A5p�C�\                                    Bxx2b  �          @�
=��33@�=q��
=���C
���33@�=q?���A�RC
�                                    BxxA  �          @�=q���\@�\)���H����C	����\@��
?��A{
=CL�                                    BxxO�  �          @�=q��ff@��<#�
=���C
=��ff@u@
�HA��
C�
                                    Bxx^T  �          @�p���z�@�p�?\)@�Q�C)��z�@s33@1G�A�z�C�
                                    Bxxl�  �          @����\)@�33����s�
C\)��\)@�@�A�\)C��                                    Bxx{�  �          @�{���H@�  �����(Q�C�����H@��@�A�
=C�H                                    Bxx�F  �          @�(���{@�G������L��C#���{@���?��AuC��                                    Bxx��  �          @�\����@�����ff���C�����@�Q�?�Q�A33CO\                                    Bxx��  �          @�����=q@�p��aG���ffC�{��=q@}p�?�(�A���C��                                    Bxx�8  �          @�{���@��þ�
=�X��Cٚ���@���?���Ao33C@                                     Bxx��  �          @�ff���R@�=q������\C����R@��?˅AL��C��                                    Bxxӄ  �          @�ff��G�@����
=��\C�f��G�@��?�  @�{CW
                                    Bxx�*  �          @�
=����@�ff��G�� z�C�����@��?�Q�A(�C:�                                    Bxx��  �          @�\����@�ff��\��z�C�����@��>�z�@�C	�{                                    Bxx�v  �          @�  ���@=p������fG�B�{���@����Vff��Q�B�=q                                    Bxx  �          @أ׿���@���  G�B�(�����@��������B�8R                                    Bxx�  �          @�
=���R@.�R��ff�y�B�ff���R@����u��33B�k�                                    Bxx+h  �          @�\)��{@(������~��B�3��{@�  �z=q�(�B��                                    Bxx:  �          @����@C33��{�b�B�\)���@��H�N{��\)B��
                                    BxxH�  �          @�Q��U�@��������Cc��U�@��
��
=�f�\B�L�                                    BxxWZ  �          @�{�p�@n�R�����>z�B����p�@�p��=q����B��                                    Bxxf   �          @�ff��@aG������P33B����@�p��5����B�                                    Bxxt�  �          @�G���
@]p�����Q�RB��\��
@��?\)�ϙ�B�L�                                    Bxx�L  �          @�(��Mp�?&ff�����m\)C(�\�Mp�@S33��\)�0��C0�                                    Bxx��  �          @�(��8��?�����Q��r�HCaH�8��@�������!�
B�#�                                    Bxx��  �          @Ӆ�3�
?ٙ���ff�p33C���3�
@�����33�z�B�G�                                    Bxx�>  �          @�G�����>���  �Hp�C-������@8Q�����z�C=q                                    Bxx��  �          @�=q�c�
?B�\���R�aG�C'�3�c�
@W�����&��C��                                    Bxx̊  �          @Ϯ�L(�?�  �����k�HC"� �L(�@hQ���=q�'33CB�                                    Bxx�0  �          @�\)�R�\>���G��l=qC+��R�\@J�H��=q�4  C                                    Bxx��  �          @�
=�N{?������R�h(�C �)�N{@l�����!�RC�q                                    Bxx�|  �          @�{�%?�G���  �}�RC��%@|������*��B�aH                                    Bxx"  �          @�p��Vff?�ff����_��C�=�Vff@qG��}p���\C�{                                    Bxx�  �          @�ff�W
=?=p������c=qC'���W
=@QG���ff�'=qC��                                    Bxx$n  �          @�\)�J�H?��H���R�e�\CT{�J�H@\)��Q���C ��                                    Bxx3  �          @�\)�333?�=q�����k�C��333@��H�w��z�B�Ǯ                                    BxxA�  �          @У׿�p�?�{���
C�=��p�@�=q���R� �HB�Ǯ                                    BxxP`  �          @�Q���
?�(�����33C����
@�  ��33�'��B�33                                    Bxx_  �          @Ϯ��\?��H��  ffC�q��\@�\)���H�(�B��                                    Bxxm�  T          @�����
>������RC"�����
@_\)���H�T��B�u�                                    Bxx|R  �          @��H�fff��R��{��CV�
�fff@#�
��p��)B�                                    Bxx��  �          @�(���=q?8Q���\)¥�)B�#׾�=q@p  �����U{B�G�                                    Bxx��  �          @θR��?�33���R\)B�33��@�(����R�(G�B�{                                    Bxx�D  �          @��
?��@,�����B���?��@�p��c33�	33B���                                    Bxx��  �          @�
=�W
=@b�\��z��b��B���W
=@�Q��8������B��                                    BxxŐ  �          @У׿��@�
�����
B��῱�@��R�|(���RB�L�                                    Bxx�6  �          @љ��&ff?�p��ȣ�#�B�LͿ&ff@�z������0�B�                                      Bxx��  �          @ҏ\�G�@#33����N��CǮ�G�@����H����RB�k�                                    Bxx�  �          @����@.{��ff�b��C�\��@��H�U����B�33                                    Bxx (  �          @�p��5�@�G���p��'Q�B����5�@��R����zffB�                                    Bxx�  �          @��0��@��
���'=qB��=�0��@�G���G��u��B��f                                    Bxxt  T          @Ӆ�7
=@r�\�����/�B�=q�7
=@�������B�                                    Bxx,  �          @�z��=p�@l(���\)�-��C �q�=p�@�ff� ����B��                                    Bxx:�  �          @���o\)@{��Y��� \)C���o\)@���u�
{B�u�                                    BxxIf  �          @Ϯ����@�p�����z�CxR����@�  ?#�
@��HCu�                                    BxxX  �          @�(��#�
@�R��  ��B��ÿ#�
@�G��n{�B�z�                                    Bxxf�  �          @θR���
@�33�\)�%�\B�Ǯ���
@��׿�G��@(�B�                                      BxxuX  �          @�Q��=q@������)�HB�33��=q@����p��W
=B�z�                                    Bxx��  �          @�  ��Q�@0  �����t�RB�G���Q�@�
=�^�R���B��
                                    Bxx��  �          @�{�.{@^{���\�c�\B�z�.{@��6ff�ԸRB�W
                                    Bxx�J  �          @�ff>�p�?�z���{W
B��>�p�@�����Q��2ffB���                                    Bxx��  �          @�ff�E�@p���ff\B�33�E�@�p�������RBŸR                                    Bxx��  "          @������@p����HQ�B�����@���z�H��HB��f                                    Bxx�<  �          @�(���
=@<�������w(�B�(���
=@�33�S33���\B��                                    Bxx��  �          @ə�?��@Mp���
=�`��B�8R?��@���8Q���  B�33                                    Bxx�  T          @�=q?
=@�Q������G  B�Q�?
=@�(��ff��z�B��{                                    Bxx�.  �          @�{�]p�@����
=���B�z��]p�@��H>���@fffB��H                                    Bxx�  �          @��
�3�
@���6ff�أ�B�{�3�
@��H���
�8Q�B�aH                                    Bxxz  �          @����K�@��\�(Q���p�B����K�@��>8Q�?�z�B�\                                    Bxx%   �          @�p��U�@�Q��A���\)B���U�@�p������>{B��                                    Bxx3�  �          @�p��j=q@����!G�����C ��j=q@�>�?�B��                                     BxxBl  �          @�z���\)@}p�?�@��RC
����\)@I��@p�A�G�C޸                                    BxxQ  �          @�=q�{�@�(��)����33C���{�@��;B�\�ٙ�C                                     Bxx_�  �          @��
�L(�@|(��w��p�C �R�L(�@����=q�AG�B�z�                                    Bxxn^  �          @�(��AG�@`�����.�HC���AG�@��������B�\                                    Bxx}  T          @���W
=@Q������"�C���W
=@��
��ff����B�#�                                    Bxx��  �          @�p��~�R@N{�vff�C��~�R@�\)��
=�s
=C�                                    Bxx�P  �          @�Q��9��@���{���RB����9��@��Ϳ�p��/�B�aH                                    Bxx��  
�          @�Q��=p�@dz������1C���=p�@�z��ff���HB�=                                    Bxx��  �          @��
�)��@c33��p��@{B�z��)��@������z�B���                                    Bxx�B  �          @�  �z�@]p����H�NffB����z�@�=q�'���Q�B���                                    Bxx��  �          @���@s33��G��R��B��
��@��\�=q����B�W
                                    Bxx�  �          @�
=�p�@H�����R�V�B�.�p�@���8�����B�Ǯ                                    Bxx�4  �          @�{�޸R@*=q���\�qz�B�k��޸R@�(��\���
=B�u�                                    Bxx �  �          @θR��@1���ff�x\)B�=��@����_\)��HB��H                                    Bxx�  �          @�Q�.{@ff��  �\B�8R�.{@��\�}p����B�=q                                    Bxx&  �          @љ����H@33���H��B�.���H@��\��=q�33B�\                                    Bxx,�  �          @У׾�p�@�����W
B�#׾�p�@��H������RB�\)                                    Bxx;r  �          @�ff>�
=?�����\)B�>�
=@�ff���
�+=qB���                                    BxxJ  �          @θR>�  ?\�ȣ�u�B�ff>�  @�  ���
�6�RB��                                    BxxX�  �          @�>�?�����Q���B�>�@�{��z��9  B��q                                    Bxxgd  �          @�ff>�?�=q�ʏ\�HB���>�@����z��E��B�Q�                                    Bxxv
  �          @�
=����?���Å�HB��3����@��H��p��C{B��3                                    Bxx��  �          @�(����@
=���}��B�8R���@���x�����B��                                    Bxx�V  �          @�(���Q�@'�����|z�B��R��Q�@����o\)�	z�BҨ�                                    Bxx��  �          @ҏ\���@'
=��z��~\)B�W
���@�Q��n�R�
�B�\                                    Bxx��  �          @�녿��\@33��  ��B��Ϳ��\@�G��~�R��B�G�                                    Bxx�H  �          @�33���R@ ����z��~�B�{���R@�p��q��p�B�B�                                    Bxx��  �          @��H��G�@����ffW
C ����G�@��\)�33B�G�                                    Bxxܔ  �          @�33��p�@-p���p��m�RB�\)��p�@�\)�_\)� 
=B�z�                                    Bxx�:  �          @�=q��p�@(���ffC G���p�@�p��\)��Bڮ                                    Bxx��  �          @ָR��p�@
=��ff�zffC�R��p�@��\�y���z�Bޣ�                                    Bxx�  T          @�
=��
@!����
�s�C&f��
@�p��p  ��HB�\)                                    Bxx,  �          @׮���@4z���(��b��Cff���@���Y�����B�3                                    Bxx%�  �          @�\)�  @%���G��m�RC��  @��i����B���                                    Bxx4x  �          @ָR�
=q@?\)����b��B��3�
=q@�{�R�\��z�B�W
                                    BxxC  �          @�{�(��@5���Y��C��(��@�
=�Mp���=qB�                                    BxxQ�  �          @��
���@/\)����a�RCE���@���S�
��B��                                    Bxx`j  �          @�(����@!����
�g�C#����@����a�� �RB���                                    Bxxo  �          @�z����@�R��(��i  C�{���@����c�
�=qB�                                      Bxx}�  �          @�  �aG�@:�H�����>�
CJ=�aG�@��H�4z���{B�\)                                    Bxx�\  �          @�  �z=q@S33��(��"(�C���z=q@��\�z����B�L�                                    Bxx�  �          @�
=�N{@>{���\�D��C	Q��N{@����5��z�B�                                      Bxx��  �          @�G��W
=@E������>�C	s3�W
=@�
=�.�R���B��=                                    Bxx�N  T          @��H�_\)@X����G��2{C�{�_\)@��
�����\B��                                    Bxx��  �          @�G��HQ�@E����\�D�Cz��HQ�@�  �2�\���B��                                    Bxx՚  �          @أ��:�H@3�
��p��T�HC\�:�H@�{�Mp����B��                                    Bxx�@  �          @ڏ\�7�@K�����Lp�C)�7�@�ff�;���p�B�\                                    Bxx��  �          @�=q�H��@{�����XG�C�{�H��@�ff�^{����B���                                    Bxx�  �          @����Mp�@{��G��[p�CaH�Mp�@�  �g��(�B�#�                                    Bxx2  �          @�Q��I��@33��Q��Z�RC���I��@����c33����B���                                    Bxx�  �          @��=p�@a����\�3ffC��=p�@�(������\B��3                                    Bxx-~  �          @ٙ��q�@����fff�  Cu��q�@��׿fff��z�B���                                    Bxx<$  �          @�Q�����@Y���|���\)C������@�ff��33�aC�H                                    BxxJ�  �          @�
=���@���  ��C5����@���������C	33                                    BxxYp  �          @�=q��Q�@:�H����+�\C����Q�@��
�����{Cn                                    Bxxh  �          @�=q�O\)@�\)�}p���B��R�O\)@�(��������B��                                    Bxxv�  T          @�33�H��@����(��p�B����H��@��Ϳ��
�-�B�
=                                    Bxx�b  �          @�Q��S33@p  �����(��CT{�S33@�G����H����B�=                                    Bxx�  �          @�  �E�@r�\����.�\C{�E�@����33���B�8R                                    Bxx��  �          @�Q��<��@^�R���<��CB��<��@����(���z�B�=q                                    Bxx�T  �          @ٙ��5�@^�R��33�BC&f�5�@�33�%���B陚                                    Bxx��  �          @ٙ��<��@L�����G�
C��<��@���333�îB�B�                                    BxxΠ  �          @���[�@tz����R�#�C޸�[�@�=q��{�~ffB�33                                    Bxx�F  �          @أ��G�@vff��33�+p�C �3�G�@�p���(���G�B홚                                    Bxx��  T          @ٙ��333@Z�H����D�\CJ=�333@���(Q����B�p�                                    Bxx��  �          @����9��@P  ��{�G��C���9��@��R�1����B��f                                    Bxx	8  �          @ٙ��'�@R�\��=q�N��C �=�'�@����7��ȏ\B�                                    Bxx�  �          @أ��1G�@   ���H�n�\C@ �1G�@��R�~�R�G�B���                                    Bxx&�  �          @أ��2�\@	�������j��Cp��2�\@���w
=�
��B�=q                                    Bxx5*  T          @�G��'�@Q������j
=C	���'�@�Q��n�R���B�33                                    BxxC�  T          @ٙ��%@(���G��iC��%@�=q�mp����B��                                    BxxRv  �          @����A�@+���p��U
=C
}q�A�@��\�QG��癚B�{                                    Bxxa  �          @ٙ��<(�@4z���ff�T�HC#��<(�@�
=�N{��ffB�R                                    Bxxo�  �          @�  �.�R@:=q��p��V��C+��.�R@����J=q���
B��                                    Bxx~h  �          @�\)�:=q@,�����W(�C	5��:=q@�33�QG���RB�z�                                    Bxx�  T          @�\)�H��@G
=��=q�CCT{�H��@����0������B�                                    Bxx��  �          @ָR��R@L����G��R(�B�aH��R@��R�8����B�                                    Bxx�Z  �          @�p��_\)@G����L��C�q�_\)@�33�Q���\B�W
                                    Bxx�   �          @�z��>�R@HQ������E33C���>�R@�Q��-p�����B�\                                    BxxǦ  �          @�p��.�R@XQ�����EG�C ���.�R@�  �'
=���
B��H                                    Bxx�L  �          @��*=q@P������K�C+��*=q@�ff�0  ��G�B�                                    Bxx��  �          @�ff�B�\@:=q���R�L\)CG��B�\@�p��>�R��\)B��                                    Bxx�  T          @�\)�XQ�@2�\��33�Ep�C���XQ�@�Q��<����(�B�
=                                    Bxx >  �          @�{�Mp�@5���z��HC
�{�Mp�@�=q�=p���B��3                                    Bxx �  �          @�ff�P��@g
=���H�-p�C��P��@�ff�����B�                                    Bxx �  �          @����\(�@��H�L�����B��q�\(�@��������RB�Q�                                    Bxx .0  �          @����u@���)����33C Q��u@���>aG�?��B��\                                    Bxx <�  �          @�\)�j=q@���Z�H����C33�j=q@�33�!G����B�B�                                    Bxx K|  �          @ָR�j=q@�Q��i���(�C��j=q@��ÿs33�ffB�                                    Bxx Z"  �          @���u�@s33�xQ����C+��u�@�Q쿰���@  B��                                    Bxx h�  �          @�p���=q@\(���Q���HC�\��=q@�Q��
=�iG�C�                                    Bxx wn  �          @��
�z=q@`  �����C
:��z=q@�녿�z��h(�B�k�                                    Bxx �  �          @��\)@W�����=qC�\�\)@��ÿ���=qC n                                    Bxx ��  �          @Ӆ��Q�@P����(���C����Q�@�p�������C33                                    Bxx �`  �          @�=q��G�@r�\�^�R� �RCǮ��G�@�G���ff�\)C ��                                    Bxx �  T          @�{��p�@�ff���h��C@ ��p�@�z�?xQ�A=qC�                                    Bxx ��  �          @�z��l(�@�p��
=��G�C O\�l(�@�
=>�Q�@P��B��                                     Bxx �R  �          @����\)@�{�|����B���\)@����}p��	�B�R                                    Bxx ��  �          @����c33@��C33��33B�Q��c33@��\�aG���\)B��f                                    Bxx �  �          @�p���ff@�33�Q���p�C\)��ff@�p�>��R@,(�C�                                    Bxx �D  �          @�ff�{�@�p��.�R��ffC��{�@��=L��>�ffB�\                                    Bxx!	�  �          @�{�C�
@���p  �z�B�{�C�
@�(��\(����
B��                                    Bxx!�  �          @�Q��ff@��\�'���p�Bۣ��ff@�33?:�H@ȣ�B؞�                                    Bxx!'6  �          @أ׿�33@����]p���ffB�.��33@�G�����ffB�ff                                    Bxx!5�  �          @�G��|��@�=q?��A7\)B�8R�|��@xQ�@xQ�B��C��                                    Bxx!D�  �          @�Q��`��@�33?�33A?
=B�{�`��@�33@�=qB�C�\                                    Bxx!S(  �          @׮�N{@��\?��\A\)B����N{@�\)@w
=B��B�W
                                    Bxx!a�  �          @ָR�>{@�p�?��
Ap�B�Q��>{@��@z=qB�B�33                                    Bxx!pt  �          @�
=�/\)@���?�  A	B��
�/\)@�p�@|��BffB�Ǯ                                    Bxx!  �          @ָR��@���?�A�
=Bڙ���@�
=@��B4p�B�Q�                                    Bxx!��  �          @�
=�@�\)?���A>{B�{�@�z�@��
B"��B�k�                                    Bxx!�f  �          @ָR�,��@�
=?���A=�B䙚�,��@�@��B  B��                                    Bxx!�  �          @�\)�,��@���?�{A(�B�.�,��@�(�@���B{B��\                                    Bxx!��  �          @�
=�8��@���>�p�@K�B�  �8��@�Q�@Z=qA�Q�B��)                                    Bxx!�X  �          @ָR�C�
@��R>#�
?�33B�Q��C�
@�G�@L(�A��B�                                     Bxx!��  T          @�{�Z�H@�\)�8Q���
B���Z�H@���@1G�A�z�B��=                                    Bxx!�  �          @Ӆ�u@����Q�����C T{�u@�p�?Q�@�
=B�B�                                    Bxx!�J  �          @�=q���@����:=q���HC{���@�녾Ǯ�W
=Cu�                                    Bxx"�  �          @��H����@���1G����C\)����@�ff������B�\)                                    Bxx"�  �          @��n{@�
=���
�x��B��H�n{@���?���A��B�{                                    Bxx" <  �          @��Tz�@��׿����z�B����Tz�@���@�A�z�B�k�                                    Bxx".�  	�          @��[�@�ff>���@%B�{�[�@�  @J�HA��B��                                    Bxx"=�  
�          @�
=�*=q@��\?�A�33B�
=�*=q@�G�@��
B.�
B��R                                    Bxx"L.  T          @����;�@�p�?�
=AB�HB���;�@��
@�  B��B��3                                    Bxx"Z�  T          @����C�
@�  >�  @B�\�C�
@�G�@QG�A�(�B�\                                    Bxx"iz  
�          @�  �e�@�����p�B�ff�e�@���@��A��\B��=                                    Bxx"x   �          @����l(�@�녿�33���B�#��l(�@�(�?�A|(�B���                                    Bxx"��  T          @����`��@��ÿ�  �p(�B��
�`��@���?�ffA1p�B�3                                    Bxx"�l  "          @�  �fff@�������{B�u��fff@��
?fff@�ffB�W
                                    Bxx"�  "          @����`��@���Y����33B�� �`��@���?��RA��B���                                    Bxx"��  �          @�33�l��@�  >��R@#33B��=�l��@���@L(�A�{B�Q�                                    Bxx"�^  T          @���j=q@�\)>��@
=qB�{�j=q@���@H��AۅB��\                                    Bxx"�  
�          @�G��W�@�G���Q�E�B�z��W�@���@7�Aʏ\B��3                                    Bxx"ު  T          @�\)�;�@��
�����5�B�(��;�@�
=?���A�{B�L�                                    Bxx"�P  
�          @�z��Mp�@��
������=qB�(��Mp�@�33?�G�A�RB��                                    Bxx"��  
�          @�  �8��@����z�H�z�B��8��@�{��Q��)G�B�Ǯ                                    Bxx#
�  �          @�Q��5@�{�\)��B�\)�5@�(�����;�B�{                                    Bxx#B  
Z          @��6ff@�Q�?�Q�A�{B��
�6ff@�{@�
=B+B��                                    Bxx#'�  �          @�{�h��@�=q?�ffA�B�{�h��@�\)@w
=B{C�                                    Bxx#6�  �          @�
=�u�@���?B�\@ǮB���u�@��H@e�A�\)C�{                                    Bxx#E4  
�          @�\)�`  @�ff?z�HAG�B��`  @�(�@w
=Bz�B�B�                                    Bxx#S�  �          @�ff��(�@���?p��@�G�C����(�@��\@_\)A��
C	
=                                    